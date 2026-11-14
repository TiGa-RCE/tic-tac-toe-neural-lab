"""Continue solver distillation to a certified finite-domain neural policy.

Run through the supported local memory guard.  This deliberately promotes the
whole ordinary 3x3 game to TRAIN; it makes no held-out generalization claim.
The browser receives only a frozen dense network, never these teacher labels.
"""

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import traceback

from train import (
    ARCH, PARAMETERS, Adam, Experiment, Network, atomic_json, digest, encode,
    gradient_check, np, softmax,
)
from solver import EMPTY, canonical_key, enumerate_states, legal_moves, play, solve, winner


BRANCH_SEEDS = (11, 23, 37)
UPDATES = 512
MAX_PHASES = 12
LEARNING_RATE = .001


class Continuation(Experiment):
    def __init__(self, run_dir, parent, seed=42):
        run_dir = Path(run_dir)
        if (run_dir / 'protocol-before-training.json').exists():
            raise ValueError('An existing continuation protocol is immutable; choose a new run directory.')
        self.parent_path = Path(parent).resolve()
        if not self.parent_path.is_file():
            raise ValueError('The selected parent NPZ checkpoint does not exist.')
        super().__init__(run_dir, seed)
        self.state.update(
            training_mode='finite-domain-continuation',
            selection_scope='full-domain TRAIN',
            recipe_note=(
                'Boundaries recipe continued from the selected student: native collection → '
                'targeted review + balanced retention with parent-correct KL → frozen qualification. '
                'All 4,520 decision states are now TRAIN. This is finite-domain mastery, '
                'not a fresh held-out generalization result. Gameplay uses only neural tensors.'
            ),
            phases=[
                self.phase_row('teacher', 'Admit the complete finite game',
                               'All 4,520 decision positions become TRAIN; original split evidence is preserved.'),
                self.phase_row('parent', 'Load the selected parent',
                               'Continue the original 6,537-parameter student; cache only parent-correct retention targets.'),
                *[self.phase_row(f'phase{i}', f'{i:02d} · Collect, repair & retain',
                                 'Three seeded branches; native rollouts, literal schedules, 512 Adam updates and external diagnosis.')
                  for i in range(1, MAX_PHASES + 1)],
                self.phase_row('selection', 'Freeze the neural policy',
                               'Rank full-domain no-loss coverage and optimal decisions; retain earlier candidates.'),
                self.phase_row('qualification', 'Independent exhaustive qualification',
                               'Grade all reachable positions and every possible opponent continuation as X and O.'),
                self.phase_row('export', 'Publish the frozen student',
                               'Export only neural tensors; compare browser logits and actions at fixed tolerance.'),
            ],
        )
        self.flush()

    @staticmethod
    def phase_row(identifier, title, detail):
        return dict(id=identifier, title=title, detail=detail, status='pending', progress=0., metrics={})

    def assess(self, net):
        """Training diagnostics over cached neural choices; no solver action override."""
        logits = net.forward(self.x)[0]
        choices = np.where(self.legal, logits, -np.inf).argmax(1)
        good = self.optimal[np.arange(len(self.boards)), choices]
        has_suboptimal = (self.legal & ~self.optimal).any(1)
        optimal_best = np.where(self.optimal, logits, -np.inf).max(1)
        suboptimal_best = np.where(self.legal & ~self.optimal, logits, -np.inf).max(1)
        minimum_margin = float((optimal_best - suboptimal_best)[has_suboptimal].min()) if has_suboptimal.any() else None

        def adversary(neural_mark):
            @lru_cache(maxsize=None)
            def visit(board):
                result = winner(board)
                if result is not None:
                    return result * neural_mark, ()
                turn = 1 if board.count(1) == board.count(-1) else -1
                if turn == neural_mark:
                    move = int(choices[self.lookup[board]])
                    value, path = visit(play(board, move))
                    return value, (move,) + path
                options = [(visit(play(board, move)), move) for move in legal_moves(board)]
                (value, path), move = min(options, key=lambda item: item[0][0])
                return value, (move,) + path

            value, moves = visit(EMPTY)
            board = EMPTY
            critical = []
            for move in moves:
                turn = 1 if board.count(1) == board.count(-1) else -1
                if turn == neural_mark:
                    index = self.lookup[board]
                    if not good[index]:
                        critical.append(index)
                board = play(board, move)
            return {'value': int(value), 'result': {-1: 'loss', 0: 'draw', 1: 'win'}[value],
                    'worst_case_moves': list(moves), 'critical_training_states': critical}

        as_x, as_o = adversary(1), adversary(-1)
        error_ids = np.flatnonzero(~good)
        return dict(
            choices=choices, good=good, errors=error_ids,
            correct=int(good.sum()), count=len(good), accuracy=float(good.mean()),
            as_x=as_x, as_o=as_o,
            no_loss_from_empty=as_x['value'] >= 0 and as_o['value'] >= 0,
            critical=sorted(set(as_x['critical_training_states'] + as_o['critical_training_states'])),
            minimum_optimal_vs_suboptimal_logit_margin=minimum_margin,
        )

    def candidate(self, net, name, branch, phase):
        result = self.assess(net)
        good = result['good']
        row = dict(
            checkpoint=name, order=len(self.candidates), branch=branch, phase=phase,
            training_correct=result['correct'], training_count=result['count'],
            # Legacy table columns are aliases; training_mode declares their actual scope.
            validation_correct=result['correct'], validation_count=result['count'],
            selection_scope='full-domain TRAIN',
            repairs=int((good & ~self.parent_good).sum()),
            breaks=int((~good & self.parent_good).sum()),
            no_loss_from_empty=result['no_loss_from_empty'],
            as_x=result['as_x'], as_o=result['as_o'],
            perfect_policy=result['correct'] == result['count'],
            minimum_optimal_vs_suboptimal_logit_margin=result['minimum_optimal_vs_suboptimal_logit_margin'],
        )
        self.candidates.append(row)
        self.state['candidates'] = self.candidates
        self.flush()
        return row, result

    @staticmethod
    def rank(candidate):
        return (candidate['no_loss_from_empty'], candidate['training_correct'],
                -candidate['breaks'], candidate['repairs'], -candidate['order'])

    def snapshot_sources(self):
        source_root = Path(__file__).resolve().parent
        paths = list(source_root.glob('*.py')) + list(source_root.glob('*.mjs')) + list((source_root / 'web').glob('*.js'))
        hashes = {}
        for path in paths:
            rel = path.relative_to(source_root)
            hashes[str(rel)] = digest(path)
            dest = self.root / 'source' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        return hashes

    def run(self):
        self.control()
        gradient_error = gradient_check()
        self.boards = [board for board in enumerate_states() if winner(board) is None]
        self.lookup = {board: index for index, board in enumerate(self.boards)}
        self.x = encode(self.boards)
        self.legal = np.array([[cell == 0 for cell in board] for board in self.boards])
        self.optimal = np.zeros((len(self.boards), 9), dtype=bool)
        for index, board in enumerate(self.boards):
            self.optimal[index, list(solve(board)[1])] = True
        all_ids = np.arange(len(self.boards))
        self.parts = dict(train=all_ids, meta=np.array([], dtype=int),
                          validation=np.array([], dtype=int), test=np.array([], dtype=int))
        keys = [canonical_key(board) for board in self.boards]
        corpus = dict(boards=self.boards, orbit_keys=keys,
                      partitions={name: ids.tolist() for name, ids in self.parts.items()},
                      orbit_membership={key: 'train' for key in sorted(set(keys))},
                      optimal_moves=[np.flatnonzero(row).tolist() for row in self.optimal],
                      scope='Every reachable decision state is deliberately supervised during continuation.')
        atomic_json(self.root / 'corpus.json', corpus)
        protocol = dict(
            training_mode='finite-domain-continuation', architecture=ARCH, parameters=PARAMETERS,
            master_seed=self.seed, branch_seeds=list(BRANCH_SEEDS),
            parent_run=self.parent_path.parent.name, parent_checkpoint=self.parent_path.name,
            parent_sha256=digest(self.parent_path), all_decision_states_promoted_to_train=True,
            previous_test_labels_used_for_training=True, held_out_generalization_claim=False,
            max_phases=MAX_PHASES, updates_per_phase=UPDATES, batch_size=128,
            branch_lr=LEARNING_RATE, adam_reset='Fresh per branch; preserved across continuation phases',
            gradient_clip=1., retention_coefficient=.1, temperature=2.,
            sampling='64 native-multiplicity samples + 64 uniform TRAIN retention; first 32 updates replace 32 native samples with current errors and worst-case critical states when present',
            native_actions='Current network greedy actions only; teacher never overrides a recorded action',
            stopping='Stop after a complete three-seed phase when any candidate is optimal on all 4,520 decision states; otherwise continue up to max_phases and require explicit qualification',
            selection='No loss from empty as both sides; then full-domain optimal count; then fewer breaks, more repairs, earliest candidate',
            corpus_sha256=digest(self.root / 'corpus.json'),
            source_hashes=self.snapshot_sources(), gradient_check_max_error=gradient_error,
        )
        atomic_json(self.root / 'protocol-before-training.json', protocol)
        self.state['dataset'] = dict(reachable=5478, decision_states=len(self.boards),
                                     symmetry_orbits=len(set(keys)), train=len(self.boards),
                                     meta=0, validation=0, test=0,
                                     split='All 4,520 states supervised; finite-domain qualification only')
        self.phase('teacher', 1., self.state['dataset'])
        self.event('All 4,520 decision states explicitly admitted as TRAIN. Old held-out evidence is preserved, not reused as a fresh test.')

        self.phase('parent')
        parent = Network(0)
        with np.load(self.parent_path, allow_pickle=False) as data:
            parent.params = [data[f'p{i}'].copy() for i in range(6)]
        if any(not np.isfinite(array).all() for array in parent.params):
            raise ValueError('Parent checkpoint contains nonfinite parameters.')
        parent_logits = parent.forward(self.x)[0]
        parent_choices = np.where(self.legal, parent_logits, -np.inf).argmax(1)
        self.parent_good = self.optimal[np.arange(len(self.boards)), parent_choices]
        self.candidates = []
        self.checkpoint(parent, Adam(parent), 'parent')
        parent_row, _ = self.candidate(parent, 'parent', None, 0)
        self.phase('parent', 1., {'correct': parent_row['training_correct'], 'count': len(self.boards)})
        self.event(f'Frozen selected parent: {parent_row["training_correct"]}/4,520 optimal; X {parent_row["as_x"]["result"]}, O {parent_row["as_o"]["result"]}.')
        branches = {seed: (parent.clone(), None, np.random.default_rng(seed ^ self.seed)) for seed in BRANCH_SEEDS}
        branches = {seed: (net, Adam(net), rng) for seed, (net, _, rng) in branches.items()}
        counts = {seed: Counter({int(index): 1 for index in all_ids}) for seed in BRANCH_SEEDS}

        completed_phase = 0
        for phase in range(1, MAX_PHASES + 1):
            phase_id = f'phase{phase}'
            self.phase(phase_id)
            for branch_index, (seed, (net, adam, rng)) in enumerate(branches.items()):
                self.control()
                native, trace = self.collect(net, rng)
                counts[seed].update(native)
                before = self.assess(net)
                target_ids = before['errors'].tolist() + before['critical'] * 4
                stem = f's{seed}-p{phase}'
                atomic_json(self.root / f'{stem}-native.json', trace)
                atomic_json(self.root / f'{stem}-diagnosis.json',
                            {key: value for key, value in before.items()
                             if key not in ('choices', 'good', 'errors')} | {'target_errors': before['errors'].tolist()})
                pool = np.array(sorted(counts[seed]))
                frequencies = np.array([counts[seed][int(index)] for index in pool], dtype=float)
                frequencies /= frequencies.sum()
                schedule = np.concatenate((rng.choice(pool, (UPDATES, 64), p=frequencies),
                                           rng.choice(all_ids, (UPDATES, 64))), axis=1)
                if target_ids:
                    schedule[:32, :32] = rng.choice(target_ids, (32, 32))
                atomic_json(self.root / f'{stem}-schedule.json',
                            dict(indices=schedule.tolist(), native_pool=pool.tolist(),
                                 multiplicity=[counts[seed][int(index)] for index in pool],
                                 target_errors=before['errors'].tolist(), critical=before['critical']))
                self.event(f'Phase {phase}, seed {seed}: saved schedule; {len(trace)} native actions, {len(before["errors"])} errors and {len(before["critical"])} adversarial critical states.')
                # Execute the literal schedule that was persisted before this phase's first update.
                literal = json.loads((self.root / f'{stem}-schedule.json').read_text())['indices']
                for update, row in enumerate(literal, 1):
                    ids = np.array(row)
                    loss, grads = net.objective(self.x[ids], self.legal[ids], self.optimal[ids],
                                               parent_logits[ids], self.parent_good[ids])
                    adam.step(net, grads, LEARNING_RATE)
                    self.step += 1
                    if update % 16 == 0:
                        self.control()
                    if update % 32 == 0:
                        self.log(net, loss, phase_id, update, seed)
                        self.phase(phase_id, min((branch_index * UPDATES + update) / (3 * UPDATES), .999),
                                   dict(seed=seed, updates=update, target_errors=len(before['errors'])))
                self.checkpoint(net, adam, stem)
                row, result = self.candidate(net, stem, seed, phase)
                self.event(f'{stem}: {result["correct"]}/4,520 optimal; repairs {row["repairs"]}, breaks {row["breaks"]}; X {result["as_x"]["result"]}, O {result["as_o"]["result"]}.')
            self.phase(phase_id, 1.)
            completed_phase = phase
            if any(candidate['perfect_policy'] for candidate in self.candidates):
                self.event(f'Complete finite-domain mastery reached after phase {phase}; no further optimizer update is needed.')
                break

        for row in self.state['phases']:
            if row['id'].startswith('phase') and int(row['id'][5:]) > completed_phase:
                row.update(status='skipped', detail='Not needed: an earlier phase reached the registered mastery condition.', progress=0.)
        self.phase('selection')
        selected = max(self.candidates, key=self.rank)
        self.state['selection'] = selected
        selected_path = self.root / f'{selected["checkpoint"]}.json'
        seal = dict(selected=selected, candidates=self.candidates,
                    model_sha256=digest(selected_path),
                    protocol_sha256=digest(self.root / 'protocol-before-training.json'),
                    frozen_at=datetime.now(timezone.utc).isoformat(),
                    selection_scope='Full finite TRAIN domain; no held-out selection claim',
                    independent_qualification_consumed=False)
        atomic_json(self.root / 'selection-before-qualification.json', seal)
        self.phase('selection', 1., {'checkpoint': selected['checkpoint'],
                                   'correct': selected['training_correct'], 'count': len(self.boards)})
        self.event(f'Frozen {selected["checkpoint"]} before independent export qualification.')

        self.phase('qualification')
        self.control()
        model = json.loads(selected_path.read_text())
        from verify_export import qualify
        independent = qualify(model, include_fixtures=True)
        atomic_json(self.root / 'independent-verification.json', independent)
        if independent['optimal_count'] != selected['training_correct']:
            raise ValueError('Independent verifier disagrees with the frozen candidate optimal count.')
        if independent['no_loss_from_empty'] != selected['no_loss_from_empty']:
            raise ValueError('Independent adversarial audit disagrees with the frozen candidate.')
        self.control()
        source_root = Path(__file__).resolve().parent
        portable = subprocess.run(
            ['node', str(source_root / 'verify_portable.mjs'), '--model', str(selected_path),
             '--fixtures', str(self.root / 'independent-verification.json'),
             '--output', str(self.root / 'portable-verification.json')], capture_output=True, text=True)
        if portable.returncode:
            raise ValueError('Browser export parity failed: ' + portable.stderr[-1000:])
        browser = json.loads((self.root / 'portable-verification.json').read_text())
        self.state['evaluation'] = dict(
            exhaustive=dict(correct=independent['optimal_count'], count=independent['total'],
                            accuracy=independent['optimal_rate'], mistakes=independent['mistake_count']),
            independent={key: independent[key] for key in ('optimal_count', 'total', 'no_loss_from_empty', 'perfect_policy')},
            worst_case=dict(as_x=independent['as_x']['result'], as_o=independent['as_o']['result']),
            browser_export=browser, selected_checkpoint=selected['checkpoint'],
            scope='All 4,520 decision positions deliberately trained. Exhaustive finite-domain correctness, not held-out generalization.',
            no_solver_fallback=True,
            minimum_optimal_vs_suboptimal_logit_margin=selected['minimum_optimal_vs_suboptimal_logit_margin'],
        )
        atomic_json(self.root / 'evaluation.json', self.state['evaluation'])
        atomic_json(self.root / 'counterexamples.json', independent['mistakes'])
        atomic_json(self.root / 'qualification-consumption.json', dict(
            selection_sha256=digest(self.root / 'selection-before-qualification.json'),
            independent_sha256=digest(self.root / 'independent-verification.json'),
            portable_sha256=digest(self.root / 'portable-verification.json'),
            consumed_at=datetime.now(timezone.utc).isoformat(),
            independent_no_loss=independent['no_loss_from_empty'],
            independent_perfect_policy=independent['perfect_policy'],
        ))
        self.phase('qualification', 1., self.state['evaluation']['exhaustive'])
        self.phase('export')
        atomic_json(self.root / 'model.json', model)
        self.state.update(model_version=selected['checkpoint'], model_available=True,
                          status='completed' if independent['no_loss_from_empty'] else 'incomplete',
                          mastery_achieved=independent['no_loss_from_empty'],
                          perfect_policy=independent['perfect_policy'])
        self.phase('export', 1., {'checkpoint': selected['checkpoint'], 'parameters': PARAMETERS})
        self.event(f'Qualification complete: {independent["optimal_count"]}/4,520 optimal; worst case X {independent["as_x"]["result"]}, O {independent["as_o"]["result"]}; browser parity passed. Solver-free neural policy exported.')
        if not independent['no_loss_from_empty']:
            self.event('Registered continuation budget ended without an unbeatable policy; status is incomplete, not successful.')
            return 2
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    experiment = Continuation(args.run_dir, args.parent, args.seed)
    try:
        return experiment.run()
    except InterruptedError as exc:
        experiment.state['status'] = 'cancelled'
        experiment.event(str(exc))
        return 0
    except Exception as exc:
        experiment.state.update(status='failed', error=str(exc))
        (experiment.root / 'traceback.txt').write_text(traceback.format_exc())
        experiment.event(f'Continuation failed: {exc}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
