"""Bounded compression search requiring optimal actions on every decision board.

Run with the supported local memory guard.  Solver labels and mistake mining
are confined to training/qualification; exported play is neural inference only.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import json
import shutil
import traceback

from compress_train import Dense, features
from train import np, Adam, atomic_json, digest, Experiment, softmax
from solver import enumerate_states, winner, solve, play

ROOT = Path(__file__).resolve().parent
DEFAULT_FAMILIES = (
    [{'architecture': [9, h, 9], 'symmetry': True}
     for h in (16, 8, 4, 2, 1, 3, 5, 6, 7, 10, 12)]
    + [{'architecture': a, 'symmetry': True} for a in (
        [9, 4, 4, 9], [9, 4, 3, 9], [9, 3, 4, 9],
        [9, 4, 2, 9], [9, 6, 4, 9], [9, 4, 6, 9], [9, 3, 3, 9],
        [18, 4, 9], [18, 8, 9], [18, 12, 9])]
    + [{'architecture': a, 'symmetry': False} for a in (
        [9, 16, 9], [9, 8, 9], [9, 4, 9], [9, 32, 9],
        [18, 8, 9], [27, 8, 9], [9, 8, 8, 9])]
)


def sizes(architecture, symmetry):
    raw = sum((a + 1) * b for a, b in zip(architecture, architecture[1:]))
    # Planned exact output-reference fold; D4 also removes six redundant biases.
    effective = raw - architecture[-2] - 1 - (6 if symmetry else 0)
    return raw, effective


def normalize_families(value):
    families = []
    for item in value:
        if isinstance(item, list):
            item = {'architecture': item, 'symmetry': True}
        architecture = item['architecture']
        if (len(architecture) < 2 or architecture[0] not in (9, 18, 27)
                or architecture[-1] != 9
                or any(type(n) is not int or not 1 <= n <= 256 for n in architecture)):
            raise ValueError('Expected bounded dense architecture with 9/18/27 inputs and 9 learned outputs.')
        if type(item.get('symmetry', True)) is not bool:
            raise ValueError('symmetry must be a boolean.')
        families.append({'architecture': list(architecture), 'symmetry': item.get('symmetry', True)})
    if not families:
        raise ValueError('At least one architecture is required.')
    return families


class PerfectSearch(Experiment):
    def __init__(self, run_dir, phases, families, seeds, updates=512,
                 kl=.001, resume=None, check_every=64):
        run_dir = Path(run_dir)
        if (run_dir / 'perfect-protocol.json').exists():
            raise ValueError('A registered search is immutable; choose a new run directory.')
        super().__init__(run_dir, 42)
        self.budget, self.families, self.seeds = phases, families, seeds
        self.updates, self.kl, self.resume = updates, kl, Path(resume) if resume else None
        self.check_every = check_every
        self.state.update(
            training_mode='finite-domain-perfect-search',
            recipe_note=(
                'Optimal-action-set teaching on all 4,520 decision boards, native collection, '
                'targeted optimality-error review and balanced retention. Safe-only targets '
                'are never substituted. The acceptance gate is 4,520/4,520 optimal under '
                'the declared deployed score rule. This is finite-domain coverage, not held-out generalization.'),
            phases=[
                dict(id='teacher', title='Admit every exact optimal-action set',
                     detail='All 4,520 decision boards are TRAIN, including forced-losing boards.',
                     status='pending', progress=0., metrics={}),
                dict(id='search', title='Search for the smallest fully optimal policy',
                     detail='Literal schedules, honest native actions, full optimality-error review and retention.',
                     status='pending', progress=0., metrics={}),
                dict(id='selection', title='Freeze a full-domain winner',
                     detail='Only 4,520/4,520 candidates qualify; independent export verification follows separately.',
                     status='pending', progress=0., metrics={}),
            ])
        self.flush()

    def assess(self, net, x, symmetry):
        logits = net.forward(x)[0]
        keys = np.floor(logits * 1e12 + .5) if symmetry else logits
        choices = np.where(self.legal, keys, -np.inf).argmax(1)
        good = self.optimal[np.arange(len(self.boards)), choices]
        optimal_best = np.where(self.optimal, logits, -np.inf).max(1)
        bad = self.legal & ~self.optimal
        has_bad = bad.any(1)
        bad_best = np.where(bad, logits, -np.inf).max(1)
        margin = float((optimal_best - bad_best)[has_bad].min())
        return dict(choices=choices, good=good, errors=np.flatnonzero(~good),
                    correct=int(good.sum()), count=len(good), accuracy=float(good.mean()),
                    minimum_optimal_vs_suboptimal_logit_margin=margin)

    def snapshot(self, net, optimizer, name, symmetry):
        arrays = {f'p{i}': p for i, p in enumerate(net.params)}
        arrays.update({f'm{i}': p for i, p in enumerate(optimizer.m)})
        arrays.update({f'v{i}': p for i, p in enumerate(optimizer.v)})
        arrays['adam_step'] = np.array(optimizer.t)
        np.savez_compressed(self.root / (name + '.npz'), **arrays)
        model = net.export(name, self.root.name)
        if symmetry:
            model['logit_round_decimals'] = 12
        model['coefficient_precision'] = 'float64'
        atomic_json(self.root / (name + '.json'), model)
        return model

    def native(self, choices, rng):
        frequency = Counter()
        trace = []
        for start in rng.choice(len(self.boards), 128, replace=True):
            board = self.boards[int(start)]
            while winner(board) is None:
                index = self.lookup[board]
                move = int(choices[index])
                frequency[index] += 1
                trace.append((index, move))
                board = play(board, move)
        return frequency, np.array(trace, dtype=np.int64)

    def run(self):
        self.phase('teacher')
        self.boards = [b for b in enumerate_states() if winner(b) is None]
        self.lookup = {b: i for i, b in enumerate(self.boards)}
        self.legal = np.array([[c == 0 for c in b] for b in self.boards])
        self.optimal = np.zeros_like(self.legal)
        for index, board in enumerate(self.boards):
            self.optimal[index, list(solve(board)[1])] = True
        corpus = dict(boards=self.boards,
                      optimal_moves=[np.flatnonzero(row).tolist() for row in self.optimal],
                      scope='All reachable decision boards deliberately used for optimal-action teaching.')
        atomic_json(self.root / 'corpus.json', corpus)
        teacher_path = ROOT / 'runs/20261113T233235Z-7e5be6/model.json'
        teacher = json.loads(teacher_path.read_text())
        parent = Dense(teacher['architecture'], 0)
        parent.params = [np.array(v) for layer in teacher['layers'] for v in (layer['weights'], layer['bias'])]
        parent_logits = parent.forward(features(self.boards, 27))[0]
        parent_choices = np.where(self.legal, parent_logits, -np.inf).argmax(1)
        parent_correct = self.optimal[np.arange(len(self.boards)), parent_choices]
        if not parent_correct.all():
            raise ValueError('The declared full-size retention teacher is not perfect.')

        resume_model = json.loads(self.resume.read_text()) if self.resume else None
        if resume_model:
            if resume_model.get('reference_logit_zero') or resume_model.get('output_bias_orbits'):
                raise ValueError('Resume uses an unfolded nine-readout training checkpoint, not a folded export.')
            if len(self.families) != 1 or self.families[0]['architecture'] != resume_model['architecture']:
                raise ValueError('Resume requires exactly its matching architecture family.')
            if (resume_model.get('symmetry_average') == 'D4') != self.families[0]['symmetry']:
                raise ValueError('Resume symmetry metadata must match the registered family.')
        source_hashes = {}
        for path in list(ROOT.glob('*.py')) + list(ROOT.glob('*.mjs')) + list((ROOT / 'web').glob('*.js')):
            relative = path.relative_to(ROOT)
            destination = self.root / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            source_hashes[str(relative)] = digest(path)
        protocol = dict(
            training_mode='finite-domain-perfect-search', families=self.families,
            seeds=self.seeds, phases_per_trial=self.budget, updates_per_phase=self.updates,
            candidate_checks_every=self.check_every, batch_size=128,
            learning_rate='0.01 phases 1-2, 0.003 phases 3-6, 0.001 thereafter',
            objective='Optimal-set CE for every phase, plus parent-correct temperature-2 KL. Never safe-set CE.',
            kl_coefficient=self.kl, temperature=2, gradient_clip=1,
            sampling='64 weighted native/current-optimality-error samples +64 uniform retention; first32updates replace32native slots with current errors when present',
            error_weight=1000, native_multiplicity_weight=20,
            stopping='Trial succeeds only at 4520/4520 rounded-greedy optimality; scheduled remaining rows then remain unexecuted and are counted explicitly.',
            deployed_parameter_order='After common-output-reference fold; D4 additionally shares two geometric output biases.',
            selection='Smallest planned deployed coefficient count among successful trials, then earliest registration.',
            parent_teacher_sha256=digest(teacher_path), corpus_sha256=digest(self.root / 'corpus.json'),
            resume_sha256=digest(self.resume) if self.resume else None,
            source_hashes=source_hashes,
            scope='Finite-domain exhaustive optimality; bounded failed training is not a global minimum proof.')
        atomic_json(self.root / 'perfect-protocol.json', protocol)
        self.state['dataset'] = dict(reachable=5478, decision_states=4520, train=4520, held_out=0)
        self.state['compression'] = dict(baseline_parameters=6537, candidates=[], trial_count=0,
                                         scope=protocol['scope'])
        self.phase('teacher', 1., self.state['dataset'])
        self.event('Exact optimal-action sets admitted for all 4,520 boards. No safe-only target relaxation is used.')

        self.phase('search')
        trials, best, best_failed = [], None, None
        for family_index, family in enumerate(self.families):
            architecture, symmetry = family['architecture'], family['symmetry']
            raw_size, effective_size = sizes(architecture, symmetry)
            if best and effective_size >= best['effective_parameters']:
                continue
            x = features(self.boards, architecture[0])
            family_best = None
            tag = '-'.join(map(str, architecture)) + ('-d4' if symmetry else '-raw')
            for seed_index, seed in enumerate(self.seeds):
                self.control()
                network = Dense(architecture, seed, symmetry)
                if resume_model:
                    network.params = [np.array(v) for layer in resume_model['layers'] for v in (layer['weights'], layer['bias'])]
                optimizer = Adam(network)
                rng = np.random.default_rng(seed)
                trial = dict(architecture=architecture, symmetry_average='D4' if symmetry else None,
                             seed=seed, raw_parameters=raw_size, effective_parameters=effective_size,
                             phases=[], best_correct=0, successful=False)
                trial_best = -1
                successful = None
                for phase in range(1, self.budget + 1):
                    self.control()
                    before = self.assess(network, x, symmetry)
                    native_counts, trace = self.native(before['choices'], rng)
                    weights = np.ones(len(self.boards))
                    for index, multiplicity in native_counts.items():
                        weights[index] += multiplicity * 20
                    weights[before['errors']] += 1000
                    schedule = np.concatenate((
                        rng.choice(len(self.boards), (self.updates, 64), p=weights / weights.sum()),
                        rng.choice(len(self.boards), (self.updates, 64))), axis=1)
                    if len(before['errors']):
                        schedule[:min(32, self.updates), :32] = rng.choice(before['errors'], (min(32, self.updates), 32))
                    stem = f'a{tag}-s{seed}-p{phase}'
                    schedule_path = self.root / (stem + '-schedule.npz')
                    np.savez_compressed(schedule_path, indices=schedule, native=trace,
                                        target_errors=before['errors'], multiplicity=weights)
                    with np.load(schedule_path, allow_pickle=False) as capsule:
                        literal = capsule['indices'].copy()
                    learning_rate = .01 if phase <= 2 else .003 if phase <= 6 else .001
                    self.event(f'{tag}, seed {seed}, phase {phase}: saved literal schedule; {len(before["errors"])} optimality errors.')
                    assessments = []
                    for update, ids in enumerate(literal, 1):
                        loss, gradients = network.objective(
                            x[ids], self.legal[ids], self.optimal[ids],
                            parent_logits[ids], parent_correct[ids], kl=self.kl)
                        optimizer.step(network, gradients, learning_rate)
                        self.step += 1
                        if update % self.check_every != 0 and update != self.updates:
                            continue
                        self.control()
                        result = self.assess(network, x, symmetry)
                        row = dict(architecture=architecture, symmetry_average='D4' if symmetry else None,
                                   forward_passes_per_move=8 if symmetry else 1,
                                   parameters=raw_size, effective_parameters=effective_size,
                                   seed=seed, phase=phase, update=update,
                                   checkpoint=stem + f'-u{update}', optimal_count=result['correct'],
                                   total=4520, perfect_policy=result['correct'] == 4520,
                                   minimum_optimal_vs_suboptimal_logit_margin=result['minimum_optimal_vs_suboptimal_logit_margin'],
                                   loss=float(loss))
                        assessments.append(row)
                        self.state['metrics'] = dict(step=self.step, phase='search', epoch=(phase-1)*self.updates+update,
                                                     branch=seed, loss=float(loss), train_accuracy=result['accuracy'], val_accuracy=None)
                        self.state['history'].append(self.state['metrics'].copy())
                        progress = (family_index + (seed_index + (phase-1+update/self.updates)/self.budget)/len(self.seeds))/len(self.families)
                        self.phase('search', min(progress, .999), dict(architecture=tag, parameters=raw_size,
                                   planned_folded_parameters=effective_size, seed=seed, phase=phase,
                                   correct=result['correct'], total=4520))
                        if result['correct'] > trial_best:
                            trial_best = result['correct']
                            self.snapshot(network, optimizer, row['checkpoint'], symmetry)
                            trial['best_correct'] = trial_best
                            trial['best_checkpoint'] = row['checkpoint']
                        if family_best is None or row['optimal_count'] > family_best['optimal_count']:
                            family_best = row.copy()
                        if best_failed is None or (row['optimal_count'], -effective_size) > (best_failed['optimal_count'], -best_failed['effective_parameters']):
                            best_failed = row.copy()
                            atomic_json(self.root / 'best-failed-so-far.json', best_failed)
                        if result['correct'] == 4520:
                            successful = row.copy()
                            trial['successful'] = True
                            break
                    trial['phases'].append(dict(phase=phase, schedule=schedule_path.name,
                        schedule_sha256=digest(schedule_path), scheduled_updates=self.updates,
                        executed_updates=update, assessments=assessments))
                    self.snapshot(network, optimizer, stem + '-phase-end', symmetry)
                    if successful:
                        break
                trials.append(trial)
                atomic_json(self.root / 'search-trials.json', trials)
                self.state['compression']['trial_count'] = len(trials)
                if successful:
                    best = successful
                    atomic_json(self.root / 'best-so-far.json', best)
                    shutil.copyfile(self.root / (best['checkpoint'] + '.json'), self.root / 'model.json')
                    self.state.update(model_available=True, model_version=best['checkpoint'])
                    self.event(f'Full optimality: {tag}, seed {seed}, {effective_size} planned folded coefficients; 4,520/4,520 under the declared score rule.')
                    break
            if family_best:
                family_best['qualification'] = 'Training-domain diagnosis; independent export audit follows separately'
                self.state['compression']['candidates'].append(family_best)
            self.flush()

        self.phase('search', 1.)
        self.phase('selection')
        self.state['selection'] = best
        if best is None:
            self.state.update(status='incomplete', best_failed=best_failed)
            self.phase('selection', 1., dict(success=False, best_correct=best_failed['optimal_count'] if best_failed else 0))
            self.event('Budget ended without full optimality. Best failed tensors and optimizer states are retained; no perfect-policy claim is made.')
            return 2
        model_path = self.root / (best['checkpoint'] + '.json')
        seal = dict(selected=best, model_sha256=digest(model_path),
                    protocol_sha256=digest(self.root / 'perfect-protocol.json'),
                    frozen_at=datetime.now(timezone.utc).isoformat(),
                    scope='Selected only after every one of 4,520 optimal-action-set checks passed; independent qualification not yet consumed.')
        atomic_json(self.root / 'selection-before-qualification.json', seal)
        shutil.copyfile(model_path, self.root / 'model.json')
        self.state.update(status='awaiting-qualification', architecture=best['architecture'],
                          parameter_count=best['parameters'], model_available=True,
                          model_version=best['checkpoint'], perfect_policy=True)
        self.state['compression'].update(selected_parameters=best['parameters'],
             selected_effective_parameters=best['effective_parameters'], selected_checkpoint=best['checkpoint'])
        self.phase('selection', 1., dict(checkpoint=best['checkpoint'], correct=4520, total=4520,
                                      planned_folded_parameters=best['effective_parameters']))
        self.event('A full-domain winner is frozen. Algebraic/precision folding and independent browser qualification remain separate required gates.')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--phases', type=int, default=12)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--check-every', type=int, default=64)
    parser.add_argument('--architectures', help='JSON list of families or architectures; bare arrays default to D4 averaging')
    parser.add_argument('--seeds', default='11,23,37')
    parser.add_argument('--kl', type=float, default=.001)
    parser.add_argument('--resume', help='Unfolded JSON checkpoint, with one matching --architectures family')
    args = parser.parse_args()
    if not 1 <= args.phases <= 256 or not 1 <= args.updates <= 4096 or not 1 <= args.check_every <= args.updates:
        parser.error('Expected bounded positive phases, updates and check interval.')
    if not 0 <= args.kl <= 1:
        parser.error('KL coefficient must be between zero and one.')
    families = normalize_families(json.loads(args.architectures) if args.architectures else DEFAULT_FAMILIES)
    seeds = [int(seed) for seed in args.seeds.split(',')]
    if not seeds or any(not 0 <= seed < 2**32 for seed in seeds):
        parser.error('Seeds must be unsigned 32-bit integers.')
    experiment = PerfectSearch(args.run_dir, args.phases, families, seeds, args.updates,
                               args.kl, args.resume, args.check_every)
    try:
        return experiment.run()
    except InterruptedError as exc:
        experiment.state['status'] = 'cancelled'
        experiment.event(str(exc))
        return 0
    except Exception as exc:
        experiment.state.update(status='failed', error=str(exc))
        (experiment.root / 'traceback.txt').write_text(traceback.format_exc())
        experiment.event(str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
