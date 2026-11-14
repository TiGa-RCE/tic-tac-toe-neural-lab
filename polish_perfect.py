"""Bounded 56-coefficient D4-policy polishing, with full-domain optimal targets.

Run through the supported memory guard. This script does not qualify an export;
it freezes any training-domain success for finalize_perfect.py to audit separately.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import shutil
import traceback

from train import np, atomic_json, digest, Experiment
from compress_train import Dense, D4, features
from solver import enumerate_states, winner, solve, play

ROOT = Path(__file__).resolve().parent
BEST_THREE = ROOT / 'runs/perfect-optimal-search-d/a9-3-9-d4-s23-p28-u224.json'
BEST_FOUR = ROOT / 'runs/perfect-optimal-search-c/a9-4-9-d4-s11-p9-u16.json'


def flatten(params):
    return np.concatenate([p.reshape(-1) for p in params])


def assign(net, vector):
    offset = 0
    for parameter in net.params:
        parameter[:] = vector[offset:offset + parameter.size].reshape(parameter.shape)
        offset += parameter.size
    if offset != len(vector):
        raise ValueError('Parameter vector dimensions do not match the network.')


def masked_distribution(logits, mask, temperature=1.):
    scaled = np.where(mask, logits / temperature, -np.inf)
    maximum = scaled.max(1, keepdims=True)
    exp = np.where(mask, np.exp(scaled - maximum), 0.)
    normalizer = exp.sum(1, keepdims=True)
    probabilities = exp / normalizer
    lse = (maximum[:, 0] + np.log(normalizer[:, 0])) * temperature
    return probabilities, lse


def objective(net, x, legal, optimal, weights, temperature, margin, ce_weight=.1, l2=1e-8):
    """Smooth max-bad versus max-optimal margin plus ordinary optimal-set CE."""
    logits, capsule = net.forward(x)
    all_p, legal_lse = masked_distribution(logits, legal)
    good_p, good_lse = masked_distribution(logits, optimal)
    delta = ce_weight * weights[:, None] * (all_p - good_p)
    loss = float(ce_weight * np.dot(weights, legal_lse - good_lse))
    bad = legal & ~optimal
    ids = np.flatnonzero(bad.any(1))
    bad_p, bad_lse = masked_distribution(logits[ids], bad[ids], temperature)
    opt_p, opt_lse = masked_distribution(logits[ids], optimal[ids], temperature)
    gaps = (bad_lse - opt_lse + margin) / temperature
    loss += float(np.dot(weights[ids], np.logaddexp(0., gaps)) * temperature)
    sigmoid = np.exp(-np.logaddexp(0., -gaps))
    delta[ids] += (weights[ids] * sigmoid)[:, None] * (bad_p - opt_p)
    # Backpropagate through the same fixed D4 average as Dense.forward.
    routed = []
    for perm in D4:
        block = np.empty_like(delta)
        block[:, perm] = delta / 8
        routed.append(block)
    delta = np.concatenate(routed)
    acts = capsule['acts']
    gradients = [None] * len(net.params)
    for layer in range(len(net.params) // 2 - 1, -1, -1):
        gradients[2 * layer] = acts[layer].T @ delta
        gradients[2 * layer + 1] = delta.sum(0)
        if layer:
            delta = (delta @ net.params[2 * layer].T) * (1 - acts[layer] ** 2)
    loss += .5 * l2 * sum(float((p * p).sum()) for p in net.params)
    gradients = [g + l2 * p for g, p in zip(gradients, net.params)]
    return loss, flatten(gradients)


def gradient_check(net, x, legal, optimal):
    """Finite differences inside the guarded run, before any optimization."""
    ids = np.flatnonzero((legal & ~optimal).any(1))[:8]
    weights = np.full(len(ids), 1. / len(ids))
    initial = flatten(net.params)
    _, gradient = objective(net, x[ids], legal[ids], optimal[ids], weights, .2, .02)
    comparisons = []
    try:
        for index in (0, 9, 26, 29, 40, 65):
            plus, minus = initial.copy(), initial.copy()
            epsilon = 1e-5
            plus[index] += epsilon
            minus[index] -= epsilon
            assign(net, plus)
            high = objective(net, x[ids], legal[ids], optimal[ids], weights, .2, .02)[0]
            assign(net, minus)
            low = objective(net, x[ids], legal[ids], optimal[ids], weights, .2, .02)[0]
            numerical = (high - low) / (2 * epsilon)
            comparisons.append(dict(index=index, analytic=float(gradient[index]), numerical=float(numerical),
                                    absolute_error=float(abs(gradient[index] - numerical))))
    finally:
        assign(net, initial)
    maximum = max(row['absolute_error'] for row in comparisons)
    return dict(maximum_absolute_error=maximum, tolerance=5e-6, passed=maximum < 5e-6, comparisons=comparisons)


class FullSuccess(Exception):
    pass


class Polishing(Experiment):
    def __init__(self, directory, phases, iterations, starts):
        directory = Path(directory)
        if directory.exists():
            raise ValueError('Choose a new run directory; previous evidence is immutable.')
        super().__init__(directory, 42)
        self.budget = phases
        self.iterations = iterations
        self.starts = starts
        self.state.update(training_mode='finite-domain-perfect-polishing', phases=[
            dict(id='teacher', title='Admit exact optimal-action sets', detail='Every reachable decision board is TRAIN.', status='pending', progress=0, metrics={}),
            dict(id='polish', title='Polish smaller neural policies', detail='Native collection, frozen error review and full-domain retention.', status='pending', progress=0, metrics={}),
            dict(id='selection', title='Freeze an all-board winner', detail='Independent export qualification remains separate.', status='pending', progress=0, metrics={})])

    def assess(self, net):
        logits = net.forward(self.x)[0]
        keys = np.floor(logits * 1e12 + .5)
        choices = np.where(self.legal, keys, -np.inf).argmax(1)
        good = self.optimal[np.arange(len(choices)), choices]
        has_bad = (self.legal & ~self.optimal).any(1)
        good_best = np.where(self.optimal, logits, -np.inf).max(1)
        bad_best = np.where(self.legal & ~self.optimal, logits, -np.inf).max(1)
        return dict(choices=choices, errors=np.flatnonzero(~good), correct=int(good.sum()),
                    margin=float((good_best - bad_best)[has_bad].min()))

    def snapshot(self, net, stem):
        model = net.export(stem, self.root.name)
        model.update(logit_round_decimals=12, coefficient_precision='float64')
        atomic_json(self.root / (stem + '.json'), model)
        np.savez_compressed(self.root / (stem + '.npz'), **{f'p{i}': p for i, p in enumerate(net.params)})

    def run(self):
        import scipy
        from scipy.optimize import minimize
        self.phase('teacher')
        self.boards = [b for b in enumerate_states() if winner(b) is None]
        self.lookup = {b: i for i, b in enumerate(self.boards)}
        self.x = features(self.boards, 9)
        self.legal = np.array([[cell == 0 for cell in board] for board in self.boards])
        self.optimal = np.zeros_like(self.legal)
        for i, board in enumerate(self.boards):
            self.optimal[i, list(solve(board)[1])] = True
        atomic_json(self.root / 'corpus.json', dict(boards=self.boards,
            optimal_moves=[np.flatnonzero(r).tolist() for r in self.optimal],
            scope='All 4,520 decision boards deliberately taught, including forced-losing positions.'))
        start_models = []
        if 'saved-three' in self.starts:
            start_models.append(('saved-three', json.loads(BEST_THREE.read_text())))
        if 'drop-four' in self.starts:
            four = json.loads(BEST_FOUR.read_text())
            for omitted in range(4):
                keep = [i for i in range(4) if i != omitted]
                model = json.loads(json.dumps(four))
                model['architecture'] = [9, 3, 9]
                model['layers'][0]['weights'] = np.array(four['layers'][0]['weights'])[:, keep].tolist()
                model['layers'][0]['bias'] = np.array(four['layers'][0]['bias'])[keep].tolist()
                model['layers'][1]['weights'] = np.array(four['layers'][1]['weights'])[keep].tolist()
                model['parameter_count'] = 66
                start_models.append((f'drop-four-{omitted}', model))
        sources = {}
        for file in ('polish_perfect.py', 'compress_train.py', 'train.py', 'solver.py'):
            target = self.root / 'source' / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / file, target)
            sources[file] = digest(target)
        protocol = dict(registered_at=datetime.now(timezone.utc).isoformat(),
            architecture=[9, 3, 9], raw_parameters=66, planned_folded_parameters=56,
            symmetry_average='D4', starts=[name for name, _ in start_models],
            start_sources={p.name: digest(p) for p in (BEST_THREE, BEST_FOUR)},
            phases_per_start=self.budget, maximum_lbfgs_iterations_per_phase=self.iterations,
            optimizer='SciPy L-BFGS-B, maxls=40, maxcor=15, ftol=1e-13, gtol=1e-8',
            scipy_version=scipy.__version__, numpy_version=np.__version__,
            objective='0.1 optimal-set CE plus annealed smooth max-bad vs max-optimal softplus margin; L2=1e-8; no safe-only labels or teacher action override.',
            gradient_acceptance='Six parameter coordinates checked by central finite differences at epsilon=1e-5, maximum absolute error below5e-6, before optimization.',
            margin=.02, temperatures=[.5, .2, .1, .05, .02, .01],
            sampling='Full-domain deterministic batches. Half uniform retention plus half normalized (1+20*error+0.05*native multiplicity). Weights and domain indices saved before each phase.',
            native='128 seeded arbitrary board starts; actual rounded greedy network moves until terminal.',
            stopping='All 4,520 rounded greedy actions optimal; training-only evidence, separate independent qualification required.',
            corpus_sha256=digest(self.root / 'corpus.json'), source_hashes=sources,
            scope='Finite-domain teaching and bounded search; not a proof of global minimality.')
        atomic_json(self.root / 'perfect-protocol.json', protocol)
        self.state['dataset'] = dict(reachable=5478, decision_states=4520, train=4520, held_out=0)
        self.state['compression'] = dict(baseline_parameters=6537, candidates=[], trial_count=0)
        self.phase('teacher', 1., self.state['dataset'])
        self.phase('polish')
        trials = []
        best = None
        selected = None
        for start_index, (name, model) in enumerate(start_models):
            net = Dense([9, 3, 9], 42, True)
            net.params = [np.array(v, dtype=float) for layer in model['layers'] for v in (layer['weights'], layer['bias'])]
            rng = np.random.default_rng(42 + start_index)
            self.snapshot(net, name + '-initial')
            if start_index == 0:
                check = gradient_check(net, self.x, self.legal, self.optimal)
                atomic_json(self.root / 'gradient-check.json', check)
                if not check['passed']:
                    raise ValueError('Polishing gradient finite-difference check failed.')
            trial = dict(start=name, phases=[], successful=False)
            for phase in range(1, self.budget + 1):
                self.control()
                before = self.assess(net)
                frequency = Counter()
                trace = []
                for start in rng.choice(len(self.boards), 128, replace=True):
                    board = self.boards[int(start)]
                    while winner(board) is None:
                        idx = self.lookup[board]
                        move = int(before['choices'][idx])
                        frequency[idx] += 1
                        trace.append((idx, move))
                        board = play(board, move)
                review = np.ones(len(self.boards))
                review[before['errors']] += 20
                for idx, count in frequency.items():
                    review[idx] += .05 * count
                weights = .5 / len(self.boards) + .5 * review / review.sum()
                temperature = protocol['temperatures'][min(phase - 1, 5)]
                stem = f'{name}-p{phase}'
                schedule = self.root / (stem + '-schedule.npz')
                np.savez_compressed(schedule, indices=np.arange(len(self.boards)), weights=weights,
                                    native=np.array(trace, dtype=np.int64), errors=before['errors'])
                with np.load(schedule, allow_pickle=False) as capsule:
                    ids, weights = capsule['indices'].copy(), capsule['weights'].copy()
                iteration = 0
                latest = None

                def fun(vector):
                    assign(net, vector)
                    return objective(net, self.x[ids], self.legal[ids], self.optimal[ids], weights, temperature, .02)

                def callback(vector, count_step=True):
                    nonlocal iteration, best, selected, latest
                    assign(net, vector)
                    if count_step:
                        iteration += 1
                        self.step += 1
                    measured = self.assess(net)
                    loss = float(fun(vector)[0])
                    latest = dict(architecture=[9, 3, 9], symmetry_average='D4', forward_passes_per_move=8,
                        parameters=66, effective_parameters=56, start=name, phase=phase, update=iteration,
                        checkpoint=stem + f'-i{iteration}', optimal_count=measured['correct'], total=4520,
                        perfect_policy=measured['correct'] == 4520,
                        minimum_optimal_vs_suboptimal_logit_margin=measured['margin'], loss=loss)
                    if best is None or (latest['optimal_count'], latest['minimum_optimal_vs_suboptimal_logit_margin']) > (best['optimal_count'], best['minimum_optimal_vs_suboptimal_logit_margin']):
                        best = latest.copy()
                        self.snapshot(net, latest['checkpoint'])
                        atomic_json(self.root / 'best-so-far.json', best)
                    self.state['metrics'] = dict(step=self.step, phase='polish', epoch=iteration,
                        branch=start_index, loss=loss, train_accuracy=measured['correct'] / 4520, val_accuracy=None)
                    self.state['history'].append(self.state['metrics'].copy())
                    self.phase('polish', (start_index + (phase - 1 + iteration / self.iterations) / self.budget) / len(start_models),
                        dict(start=name, phase=phase, correct=measured['correct'], total=4520, temperature=temperature))
                    if measured['correct'] == 4520:
                        selected = latest.copy()
                        self.snapshot(net, selected['checkpoint'])
                        raise FullSuccess()

                result = None
                try:
                    result = minimize(fun, flatten(net.params), method='L-BFGS-B', jac=True,
                        callback=callback, options=dict(maxiter=self.iterations, maxls=40, maxcor=15, ftol=1e-13, gtol=1e-8))
                    assign(net, result.x)
                    # Even a zero-iteration optimizer endpoint is assessed and saved.
                    if iteration == 0:
                        callback(result.x, count_step=False)
                except FullSuccess:
                    pass
                self.snapshot(net, stem + '-end')
                if result is not None:
                    state = dict(x=result.x, gradient=result.jac)
                    if hasattr(result.hess_inv, 'sk') and hasattr(result.hess_inv, 'yk'):
                        state.update(lbfgs_sk=result.hess_inv.sk, lbfgs_yk=result.hess_inv.yk)
                    np.savez_compressed(self.root / (stem + '-optimizer.npz'), **state)
                trial['phases'].append(dict(phase=phase, temperature=temperature, schedule=schedule.name,
                    schedule_sha256=digest(schedule), optimizer_iterations=iteration,
                    optimizer_success=bool(result.success) if result is not None else None,
                    optimizer_message=str(result.message) if result is not None else 'full-domain success interrupt',
                    last_assessment=latest))
                atomic_json(self.root / 'search-trials.json', trials + [trial])
                if selected:
                    trial['successful'] = True
                    break
            trials.append(trial)
            self.state['compression']['trial_count'] = len(trials)
            self.state['compression']['candidates'].append(trial['phases'][-1]['last_assessment'])
            atomic_json(self.root / 'search-trials.json', trials)
            if selected:
                break
        self.phase('polish', 1.)
        self.phase('selection')
        self.state['selection'] = selected
        if not selected:
            self.state.update(status='incomplete', best_failed=best)
            self.phase('selection', 1., dict(success=False, best_correct=best['optimal_count'] if best else 0))
            self.event('Bounded L-BFGS polishing ended without full optimality; best and failed endpoints retained.')
            return 2
        checkpoint = self.root / (selected['checkpoint'] + '.json')
        atomic_json(self.root / 'selection-before-qualification.json', dict(selected=selected,
            model_sha256=digest(checkpoint), protocol_sha256=digest(self.root / 'perfect-protocol.json'),
            frozen_at=datetime.now(timezone.utc).isoformat(), scope='Training-only full-domain winner; independent export qualification pending.'))
        shutil.copyfile(checkpoint, self.root / 'model.json')
        self.state.update(status='awaiting-qualification', architecture=[9, 3, 9], parameter_count=66,
                          model_version=selected['checkpoint'], model_available=True)
        self.state['compression'].update(selected_parameters=66, selected_effective_parameters=56,
                                         selected_checkpoint=selected['checkpoint'])
        self.phase('selection', 1., dict(success=True, correct=4520, total=4520, planned_folded_parameters=56))
        self.event('A 56-planned-coefficient training success is frozen; independent export audit remains required.')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--phases', type=int, default=6)
    parser.add_argument('--starts', default='saved-three,drop-four')
    args = parser.parse_args()
    starts = args.starts.split(',')
    if not 1 <= args.iterations <= 2000 or not 1 <= args.phases <= 24 or not starts or any(s not in ('saved-three', 'drop-four') for s in starts):
        parser.error('Use bounded iterations/phases and saved-three and/or drop-four starts.')
    experiment = Polishing(args.run_dir, args.phases, args.iterations, starts)
    try:
        return experiment.run()
    except InterruptedError as error:
        experiment.state['status'] = 'cancelled'
        experiment.event(str(error))
        return 0
    except Exception as error:
        experiment.state.update(status='failed', error=str(error))
        (experiment.root / 'traceback.txt').write_text(traceback.format_exc())
        experiment.event(str(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
