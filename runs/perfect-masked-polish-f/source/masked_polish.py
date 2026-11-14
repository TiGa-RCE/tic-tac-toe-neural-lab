"""Bounded beam search over fixed-zero masks with full-domain L-BFGS fitting.

Run through the supported model guard. Search checkpoints are masked dense
training artifacts. Independent sparse export qualification remains required.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import shutil
import time
import traceback

from train import np, atomic_json, digest
from compress_train import D4, features
from prune_perfect import Folded
from solver import enumerate_states, winner, solve, play

ROOT = Path(__file__).resolve().parent


def free_vector(net):
    return np.concatenate([p[m] for p, m in zip(net.params, net.masks)])


def assign_free(net, vector):
    offset = 0
    for p, mask in zip(net.params, net.masks):
        count = int(mask.sum())
        p[mask] = vector[offset:offset+count]
        p[~mask] = 0.
        offset += count
    if offset != len(vector):
        raise ValueError('Free coefficient dimensions do not match the mask')


def distribution(logits, mask, temperature=1.):
    scaled = np.where(mask, logits / temperature, -np.inf)
    maximum = scaled.max(axis=1, keepdims=True)
    exp = np.where(mask, np.exp(scaled-maximum), 0.)
    denominator = exp.sum(axis=1, keepdims=True)
    return exp/denominator, (maximum[:, 0]+np.log(denominator[:, 0]))*temperature


def objective(net, x, legal, optimal, weights, temperature, margin,
              ce_weight=.05, l2=1e-9):
    logits, (all_x, hidden, _) = net.forward(x)
    legal_p, legal_lse = distribution(logits, legal)
    optimal_p, optimal_lse = distribution(logits, optimal)
    loss = float(ce_weight*np.dot(weights, legal_lse-optimal_lse))
    delta = ce_weight*weights[:, None]*(legal_p-optimal_p)
    bad = legal & ~optimal
    informative = np.flatnonzero(bad.any(axis=1))
    bad_p, bad_lse = distribution(logits[informative], bad[informative], temperature)
    good_p, good_lse = distribution(logits[informative], optimal[informative], temperature)
    gap = (bad_lse-good_lse+margin)/temperature
    sigmoid = np.exp(-np.logaddexp(0., -gap))
    loss += float(np.dot(weights[informative], np.logaddexp(0., gap))*temperature)
    delta[informative] += (weights[informative]*sigmoid)[:, None]*(bad_p-good_p)
    blocks = []
    for permutation in D4:
        block = np.empty_like(delta)
        block[:, permutation] = delta/8
        blocks.append(block[:, :8])
    routed = np.concatenate(blocks)
    out_w = hidden.T @ routed
    out_b = np.array([routed[:, [1, 3, 5, 7]].sum(), routed[:, 4].sum()])
    hidden_delta = (routed @ net.params[2].T)*(1-hidden**2)
    gradients = [all_x.T @ hidden_delta, hidden_delta.sum(axis=0), out_w, out_b]
    loss += .5*l2*sum(float((p[m]*p[m]).sum()) for p, m in zip(net.params, net.masks))
    gradient = np.concatenate([(g+l2*p)[m] for p, g, m in zip(net.params, gradients, net.masks)])
    return loss, gradient


def gradient_check(net, x, legal, optimal):
    rows = np.flatnonzero((legal & ~optimal).any(axis=1))[:12]
    weights = np.full(len(rows), 1./len(rows))
    vector = free_vector(net)
    _, analytic = objective(net, x[rows], legal[rows], optimal[rows], weights, .03, .005)
    comparisons = []
    try:
        for index in sorted(set(np.linspace(0, len(vector)-1, min(10, len(vector)), dtype=int))):
            plus, minus = vector.copy(), vector.copy()
            epsilon = 1e-5
            plus[index] += epsilon
            minus[index] -= epsilon
            assign_free(net, plus)
            high = objective(net, x[rows], legal[rows], optimal[rows], weights, .03, .005)[0]
            assign_free(net, minus)
            low = objective(net, x[rows], legal[rows], optimal[rows], weights, .03, .005)[0]
            numerical = (high-low)/(2*epsilon)
            comparisons.append({'free_index': int(index), 'analytic': float(analytic[index]),
                                'numerical': float(numerical),
                                'absolute_error': float(abs(numerical-analytic[index]))})
    finally:
        assign_free(net, vector)
    maximum = max(row['absolute_error'] for row in comparisons)
    return {'passed': maximum < 5e-6, 'maximum_absolute_error': maximum,
            'tolerance': 5e-6, 'epsilon': 1e-5, 'comparisons': comparisons}


class CompletePolicy(Exception):
    pass


def run(args):
    import scipy
    from scipy.optimize import minimize
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    start = Path(args.start)
    base = Folded(json.loads(start.read_text()))
    boards = [b for b in enumerate_states() if winner(b) is None]
    if len(boards) != 4520:
        raise ValueError('Expected exactly 4,520 decision boards')
    lookup = {b: i for i, b in enumerate(boards)}
    x = features(boards, 9)
    legal = np.array([[cell == 0 for cell in b] for b in boards])
    optimal = np.zeros_like(legal)
    for i, board in enumerate(boards):
        optimal[i, list(solve(board)[1])] = True
    atomic_json(root/'corpus.json', {'boards': boards,
        'optimal_moves': [np.flatnonzero(row).tolist() for row in optimal],
        'scope': 'Every decision board is deliberately taught, including forced losses; no held-out claim.'})
    source_hashes = {}
    for name in ('masked_polish.py', 'prune_perfect.py', 'compress_train.py', 'train.py', 'solver.py'):
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        source_hashes[name] = digest(target)
    temperatures = [.05, .01, .002, .0005, .0001]
    protocol = {'registered_at': datetime.now(timezone.utc).isoformat(),
        'start_model': str(start), 'start_model_sha256': digest(start),
        'source_hashes': source_hashes, 'corpus_sha256': digest(root/'corpus.json'),
        'seeds': args.seeds, 'maximum_prune_depth': args.steps,
        'beam_width': args.beam_width, 'candidate_masks_per_parent_per_seed': args.attempts,
        'maximum_phases_per_candidate': args.phases, 'maximum_iterations_per_phase': args.iterations,
        'assess_every_optimizer_iterations': args.check_every,
        'objective': '0.05 optimal-set CE plus annealed smooth max-bad versus max-optimal softplus margin; masked free coordinates only; L2 1e-9.',
        'target_margin': args.margin, 'temperatures': temperatures,
        'retention': 'Every full-domain batch is half uniform plus half normalized review weights1+1000*current_error+20*native_multiplicity.',
        'native': '128 seeded starts; actual current rounded greedy actions until terminal, no teacher override.',
        'pruning': 'Layer-RMS normalized magnitude; independent seeded rotations of the candidate rank list; keep a beam of distinct successful zero masks.',
        'optimizer': 'SciPy L-BFGS-B; maxls40,maxcor15,ftol1e-13,gtol1e-9; fresh state each registered phase.',
        'gradient_check': 'Central differences of ten free coordinates, epsilon1e-5, absolute error below5e-6 before optimization.',
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'success': 'All4520roundedgreedyactionsoptimal AND positive optimal-versus-suboptimal margin; separate sparse export qualification required.',
        'selection': 'Fewest active coefficients, then largest minimum margin, then earliest checkpoint.',
        'scope': 'Bounded full-domain solver-taught neural fitting; no runtime solver fallback or global minimum proof.'}
    atomic_json(root/'masked-polish-protocol.json', protocol)
    state = {'status': 'running', 'run_id': root.name, 'training_mode': 'finite-domain-masked-polishing',
             'parameter_count': base.count, 'trials': [], 'history': [], 'best': None}

    def flush():
        state['elapsed_seconds'] = round(time.monotonic()-began, 3)
        atomic_json(root/'status.json', state)

    def assess(net):
        logits = net.forward(x)[0]
        choices = np.where(legal, np.floor(logits*1e12+.5), -np.inf).argmax(axis=1)
        good = optimal[np.arange(len(boards)), choices]
        informative = (legal & ~optimal).any(axis=1)
        gaps = np.where(optimal, logits, -np.inf).max(axis=1)-np.where(legal & ~optimal, logits, -np.inf).max(axis=1)
        return {'optimal_count': int(good.sum()), 'total': 4520,
                'minimum_margin': float(gaps[informative].min()),
                'choices': choices, 'errors': np.flatnonzero(~good)}

    def summary(measured):
        return {k: v for k, v in measured.items() if k not in ('choices', 'errors')}

    def success(measured):
        return measured['optimal_count'] == 4520 and measured['minimum_margin'] > 0

    def snapshot(net, stem):
        model = net.export(stem, root.name)
        model.update(learned_parameter_count=net.count, effective_parameter_count=net.count)
        atomic_json(root/(stem+'.json'), model)
        arrays = {f'p{i}': p for i, p in enumerate(net.params)}
        arrays.update({f'mask{i}': m for i, m in enumerate(net.masks)})
        np.savez_compressed(root/(stem+'.npz'), **arrays)

    initial = assess(base)
    atomic_json(root/'parent-diagnosis.json', summary(initial))
    if not success(initial):
        raise ValueError('The numeric masked parent must already be fully optimal with positive margin')
    check = gradient_check(base, x, legal, optimal)
    atomic_json(root/'gradient-check.json', check)
    if not check['passed']:
        raise ValueError('Masked objective failed its finite-difference gradient check')
    snapshot(base, 'initial')
    best = None
    beam = [{'net': base, 'checkpoint': 'initial', 'margin': initial['minimum_margin']}]
    all_tried = set()
    for depth in range(1, args.steps+1):
        successes = []
        for parent_index, parent in enumerate(beam):
            ranked = sorted((abs(float(p[index]))/max(float(np.sqrt((p*p).mean())), 1e-12), k, index)
                            for k, (p, mask) in enumerate(zip(parent['net'].params, parent['net'].masks))
                            for index in np.ndindex(p.shape) if mask[index])
            for seed in args.seeds:
                # A second seed explores the next low-magnitude candidates,
                # rather than refitting the same mask with identical weights.
                offset = args.seeds.index(seed)*args.attempts
                candidates = (ranked[offset:]+ranked[:offset])[:args.attempts]
                for attempt, (saliency, layer, index) in enumerate(candidates, 1):
                    net = parent['net'].clone()
                    net.masks[layer][index] = False
                    net.project()
                    mask_key = tuple(m.tobytes() for m in net.masks)
                    if mask_key in all_tried:
                        continue
                    all_tried.add(mask_key)
                    rng = np.random.default_rng(seed+depth*1009+parent_index*100003+attempt*17)
                    stem = f'd{depth}-b{parent_index}-s{seed}-a{attempt}'
                    trial = {'depth': depth, 'parent_checkpoint': parent['checkpoint'], 'seed': seed,
                             'attempt': attempt, 'parameters': net.count, 'pruned_layer': layer,
                             'pruned_index': list(index), 'saliency': saliency, 'phases': [],
                             'successful': False, 'optimizer_iterations': 0}
                    state['trials'].append(trial)
                    latest = assess(net)
                    best_attempt = summary(latest)
                    best_vector = free_vector(net).copy()
                    best_checkpoint = stem+'-u0'
                    snapshot(net, best_checkpoint)
                    for phase in range(1, args.phases+1):
                        if success(latest):
                            break
                        # Restart the next annealing stage from the best actual
                        # policy found so far, preserving both failures and endpoints.
                        assign_free(net, best_vector)
                        before = assess(net)
                        frequency, trace = Counter(), []
                        starts = rng.choice(len(boards), 128, replace=True)
                        for start_row in starts:
                            board = boards[int(start_row)]
                            while winner(board) is None:
                                row = lookup[board]
                                move = int(before['choices'][row])
                                frequency[row] += 1
                                trace.append((row, move))
                                board = play(board, move)
                        review = np.ones(len(boards))
                        review[before['errors']] += 1000
                        for row, multiplicity in frequency.items():
                            review[row] += 20*multiplicity
                        weights = .5/len(boards)+.5*review/review.sum()
                        schedule = root/f'{stem}-p{phase}-schedule.npz'
                        np.savez_compressed(schedule, indices=np.arange(len(boards)), weights=weights,
                                            starts=starts, native_trace=np.array(trace, dtype=np.int64),
                                            errors=before['errors'])
                        temperature = temperatures[min(phase-1, len(temperatures)-1)]
                        record = {'phase': phase, 'schedule': schedule.name,
                                  'schedule_sha256': digest(schedule), 'temperature': temperature,
                                  'before': summary(before), 'optimizer_iterations': 0}
                        trial['phases'].append(record)
                        flush()  # Registration and literal weights precede every optimizer call.
                        iteration = 0
                        result = None
                        phase_stem = f'{stem}-p{phase}'

                        def fun(vector):
                            assign_free(net, vector)
                            return objective(net, x, legal, optimal, weights, temperature, args.margin)

                        def observe(vector, count_step=True):
                            nonlocal iteration, latest, best_attempt, best_vector, best_checkpoint
                            assign_free(net, vector)
                            if count_step:
                                iteration += 1
                                trial['optimizer_iterations'] += 1
                            if count_step and iteration % args.check_every:
                                return
                            latest = assess(net)
                            loss = float(fun(vector)[0])
                            row = {'depth': depth, 'parent': parent_index, 'seed': seed, 'attempt': attempt,
                                   'phase': phase, 'iteration': iteration, 'parameters': net.count,
                                   'loss': loss, **summary(latest)}
                            state['history'].append(row)
                            quality = (latest['optimal_count'], latest['minimum_margin'])
                            prior_quality = (best_attempt['optimal_count'], best_attempt['minimum_margin'])
                            if quality > prior_quality:
                                best_attempt = summary(latest)
                                best_vector = free_vector(net).copy()
                                best_checkpoint = phase_stem+f'-i{iteration}'
                                snapshot(net, best_checkpoint)
                            record['optimizer_iterations'] = iteration
                            record['latest'] = summary(latest)
                            flush()
                            if success(latest):
                                raise CompletePolicy()

                        try:
                            result = minimize(fun, free_vector(net), jac=True, method='L-BFGS-B',
                                callback=observe, options={'maxiter': args.iterations, 'maxls': 40,
                                    'maxcor': 15, 'ftol': 1e-13, 'gtol': 1e-9})
                            observe(result.x, count_step=False)
                        except CompletePolicy:
                            pass
                        endpoint = free_vector(net).copy()
                        endpoint_loss, endpoint_gradient = fun(endpoint)
                        arrays = {'x': endpoint, 'gradient': endpoint_gradient}
                        if result is not None and hasattr(result.hess_inv, 'sk'):
                            arrays.update(lbfgs_sk=result.hess_inv.sk, lbfgs_yk=result.hess_inv.yk)
                        np.savez_compressed(root/(phase_stem+'-optimizer.npz'), **arrays)
                        snapshot(net, phase_stem+'-end')
                        record.update(optimizer_success=bool(result.success) if result is not None else None,
                            optimizer_message=str(result.message) if result is not None else 'full-domain success interrupt',
                            endpoint_loss=float(endpoint_loss), endpoint=summary(assess(net)))
                        # Retain the strongest actual policy, not merely the last line search.
                        assign_free(net, best_vector)
                        latest = assess(net)
                        flush()
                    trial.update(successful=success(latest), best=best_attempt,
                                 best_checkpoint=best_checkpoint, final=summary(latest))
                    if success(latest):
                        checkpoint = stem+'-accepted'
                        snapshot(net, checkpoint)
                        candidate = {'checkpoint': checkpoint, 'parameters': net.count,
                                     'planned_normalized_parameters': net.count-1,
                                     'depth': depth, 'seed': seed, **summary(latest)}
                        successes.append({'net': net.clone(), 'checkpoint': checkpoint,
                                          'margin': latest['minimum_margin'], 'mask_key': mask_key})
                        if (best is None or (candidate['parameters'], -candidate['minimum_margin'])
                                < (best['parameters'], -best['minimum_margin'])):
                            best = candidate
                            snapshot(net, 'model')
                            atomic_json(root/'selection-before-qualification.json', {
                                'selected': best, 'model_sha256': digest(root/'model.json'),
                                'source_protocol_sha256': digest(root/'masked-polish-protocol.json'),
                                'frozen_at': datetime.now(timezone.utc).isoformat(),
                                'scope': 'Training-domain success; independent sparse export qualification remains required.'})
                            state['best'] = best
                    flush()
        if not successes:
            break
        successes.sort(key=lambda candidate: -candidate['margin'])
        beam = successes[:args.beam_width]
    state['status'] = 'awaiting-qualification' if best is not None else 'incomplete'
    flush()
    atomic_json(root/'search-trials.json', state['trials'])
    print(json.dumps({'run_id': root.name, 'status': state['status'], 'best': best,
                      'trial_count': len(state['trials']), 'elapsed_seconds': state['elapsed_seconds']}))
    return 0 if best else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--start', default=str(ROOT/'runs/perfect-pruning-b/model.json'))
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--attempts', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--phases', type=int, default=3)
    parser.add_argument('--beam-width', type=int, default=2)
    parser.add_argument('--seeds', default='11,23')
    parser.add_argument('--check-every', type=int, default=5)
    parser.add_argument('--margin', type=float, default=.005)
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(',')]
    if not (1 <= args.steps <= 53 and 1 <= args.attempts <= 74
            and 1 <= args.iterations <= 1000 and 1 <= args.phases <= 8
            and 1 <= args.beam_width <= 8 and 1 <= args.check_every <= args.iterations
            and args.seeds and len(set(args.seeds)) == len(args.seeds)
            and all(0 <= seed < 2**32 for seed in args.seeds) and 0 < args.margin <= 1):
        parser.error('Expected bounded positive budgets, distinct unsigned seeds, and margin<=1.')
    if Path(args.run_dir).exists():
        parser.error('Use a new immutable run directory.')
    try:
        return run(args)
    except Exception:
        root = Path(args.run_dir)
        if root.exists():
            (root/'traceback.txt').write_text(traceback.format_exc())
            state_path = root/'status.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            state['status'] = 'failed'
            atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
