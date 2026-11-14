"""Fit a sparse D4 linear head while freezing the certified hidden features.

Run only through the supported local model-work launcher.  Dense intermediate
models retain all 74 slots; learned_parameter_mask distinguishes structural zero
slots for the final, separately qualified compact runtime representation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

from compress_train import D4, features
from solver import enumerate_states, play, solve, winner

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temp.replace(path)


def build_basis(model, boards):
    if (model['architecture'] != [9, 4, 8]
            or model.get('symmetry_average') != 'D4'
            or model.get('output_bias_orbits') != 'D4-corner-zero'
            or model.get('reference_logit_zero') is not True):
        raise ValueError('Expected the frozen 9→4→8, fixed-zero, shared-bias D4 model.')
    x = features(boards, 9)
    first = model['layers'][0]
    w, b = np.array(first['weights']), np.array(first['bias'])
    basis = np.zeros((len(boards), 9, 34))
    for perm in D4:
        transformed = np.empty_like(x)
        transformed[:, perm] = x
        hidden = np.tanh(transformed @ w + b)
        for action, raw_column in enumerate(perm):
            if raw_column < 8:
                for unit in range(4):
                    basis[:, action, unit * 8 + raw_column] += hidden[:, unit] / 8
    basis[:, [1, 3, 5, 7], 32] = 1
    basis[:, 4, 33] = 1
    head = np.r_[np.array(model['layers'][1]['weights']).ravel(),
                 np.array(model['layers'][1]['bias'])]
    return basis, head


def assess(basis, head, legal, optimal):
    logits = np.einsum('nak,k->na', basis, head)
    keys = np.floor(logits * 1e12 + .5)
    choices = np.where(legal, keys, -np.inf).argmax(axis=1)
    good = optimal[np.arange(len(choices)), choices]
    bad = legal & ~optimal
    informative = bad.any(axis=1)
    gap = (np.where(optimal, logits, -np.inf).max(axis=1)
           - np.where(bad, logits, -np.inf).max(axis=1))
    return choices, int(good.sum()), float(gap[informative].min())


def model_with_head(parent, head, active, checkpoint, run_id):
    model = json.loads(json.dumps(parent))
    head = head.copy()
    head[~active] = 0.
    model['layers'][1]['weights'] = head[:32].reshape(4, 8).tolist()
    model['layers'][1]['bias'] = head[32:].tolist()
    model.update(checkpoint=checkpoint, run_id=run_id, coefficient_precision='float64',
                 parameter_count=74, learned_parameter_count=40 + int(active.sum()),
                 effective_parameter_count=40 + int(active.sum()),
                 head_active_mask=active.tolist(),
                 learned_parameter_mask=[
                     {'weights': np.ones((9, 4), dtype=bool).tolist(), 'bias': [True] * 4},
                     {'weights': active[:32].reshape(4, 8).tolist(), 'bias': active[32:].tolist()},
                 ],
                 intermediate_format='Dense slots retained; final sparse export and independent qualification required.')
    return model


def main(args):
    began = time.monotonic()
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=False)
    model = json.loads(args.model.read_text())
    boards = [board for board in enumerate_states() if winner(board) is None]
    if len(boards) != 4520:
        raise ValueError('Expected all 4,520 reachable decision boards.')
    legal = np.array([[cell == 0 for cell in board] for board in boards])
    optimal = np.zeros_like(legal)
    for row, board in enumerate(boards):
        optimal[row, list(solve(board)[1])] = True
    basis, original_head = build_basis(model, boards)
    choices, correct, margin = assess(basis, original_head, legal, optimal)
    if correct != 4520 or margin <= 0:
        raise ValueError('Parent must be strictly optimal throughout the domain.')

    # Freeze current native policy before any LP fitting. Teacher action sets
    # grade these choices; they never replace them in the native collection.
    lookup = {board: row for row, board in enumerate(boards)}
    rng = np.random.default_rng(11)
    starts = rng.choice(len(boards), 128, replace=True)
    trace, counts = [], Counter()
    for start in starts:
        board = boards[int(start)]
        while winner(board) is None:
            index = lookup[board]
            action = int(choices[index])
            trace.append((index, action))
            counts[index] += 1
            board = play(board, action)
    np.savez_compressed(root / 'native-before-fit.npz', starts=starts,
                        trace=np.array(trace, dtype=np.int64))
    write_json(root / 'corpus.json', {'boards': boards,
               'optimal_moves': [np.flatnonzero(row).tolist() for row in optimal],
               'frozen_native_choices': choices.tolist(),
               'native_multiplicity': {str(k): v for k, v in counts.items()},
               'scope': 'All 4,520 boards are teaching data; no held-out claim.'})

    rows = []
    row_ids = []
    for index, action in enumerate(choices):
        for bad in np.flatnonzero(legal[index] & ~optimal[index]):
            rows.append(basis[index, action] - basis[index, bad])
            row_ids.append((index, int(action), int(bad)))
    constraints = np.array(rows)
    np.savez_compressed(root / 'constraints-before-fit.npz', matrix=constraints,
                        board_chosen_bad=np.array(row_ids, dtype=np.int64),
                        rhs=np.ones(len(rows)))
    source = root / 'source'
    source.mkdir()
    shutil.copyfile(__file__, source / Path(__file__).name)
    protocol = {'registered_at': datetime.now(timezone.utc).isoformat(),
                'source_model': str(args.model), 'source_model_sha256': digest(args.model),
                'source_script_sha256': digest(__file__),
                'corpus_sha256': digest(root / 'corpus.json'),
                'constraints_sha256': digest(root / 'constraints-before-fit.npz'),
                'native_trace_sha256': digest(root / 'native-before-fit.npz'),
                'hidden_coefficients_frozen': 40, 'head_slots': 34,
                'constraint_rows': len(rows), 'margin_target': 1.,
                'orders': args.orders.split(','), 'maximum_pruning_attempts_per_branch': args.max_attempts,
                'lp_time_limit_seconds': args.lp_time_limit,
                'objective': 'Minimum L1 head under every frozen optimal-versus-suboptimal margin constraint.',
                'pruning': 'Sequential exact-zero constraints; commit a removal only on feasible LP and full 4,520 greedy optimality.',
                'acceptance': 'Training-domain full optimality only; final sparse representation needs separate independent oracle and browser qualification.',
                'scope': 'Finite-domain supervised linear fitting and structural pruning; no solver fallback or global-minimum proof.'}
    write_json(root / 'sparse-head-protocol.json', protocol)

    # Variables are [head34, abs_head34]. Inactive slots are fixed to exact zero.
    identity = sparse.eye(34, format='csr')
    a_ub = sparse.vstack((sparse.hstack((-sparse.csr_matrix(constraints), sparse.csr_matrix((len(rows), 34)))),
                          sparse.hstack((identity, -identity)),
                          sparse.hstack((-identity, -identity))), format='csr')
    b_ub = np.r_[-np.ones(len(rows)), np.zeros(68)]
    trials = []
    best = None

    def fit(active, branch, attempt, removed, objective_weights):
        nonlocal best
        started = time.monotonic()
        bounds = [(None, None) if yes else (0., 0.) for yes in active]
        bounds += [(0., None) if yes else (0., 0.) for yes in active]
        result = linprog(np.r_[np.zeros(34), objective_weights], A_ub=a_ub, b_ub=b_ub,
                         bounds=bounds, method='highs',
                         options={'time_limit': args.lp_time_limit,
                                  'primal_feasibility_tolerance': 1e-8,
                                  'dual_feasibility_tolerance': 1e-8})
        record = {'branch': branch, 'attempt': attempt, 'removed_slot': removed,
                  'active_mask': active.tolist(), 'active_head_count': int(active.sum()),
                  'effective_parameter_count': 40 + int(active.sum()),
                  'solver_status': int(result.status), 'solver_message': result.message,
                  'elapsed_seconds': round(time.monotonic() - started, 6), 'accepted': False}
        head = None
        if result.success and result.x is not None:
            head = result.x[:34].copy()
            head[~active] = 0.
            actual_margin = float((constraints @ head).min())
            _, count, actual_gap = assess(basis, head, legal, optimal)
            record.update(minimum_frozen_constraint_margin=actual_margin,
                          optimal_count=count, total=4520,
                          minimum_optimal_vs_suboptimal_logit_margin=actual_gap,
                          head_l1=float(np.abs(head).sum()))
            record['accepted'] = bool(np.isfinite(head).all() and actual_margin >= 1 - 1e-6
                                      and count == 4520 and actual_gap > 0)
            if record['accepted']:
                checkpoint = f'head-{record["effective_parameter_count"]}-{branch}-{attempt}'
                record['checkpoint'] = checkpoint
                write_json(root / (checkpoint + '.json'),
                           model_with_head(model, head, active, checkpoint, root.name))
                if best is None or (record['effective_parameter_count'], record['head_l1']) < (best['effective_parameter_count'], best['head_l1']):
                    best = dict(record)
                    write_json(root / 'best-before-qualification.json', best)
        trials.append(record)
        write_json(root / 'sparse-head-trials.json', trials)
        return head if record['accepted'] else None

    for branch in protocol['orders']:
        active = np.ones(34, dtype=bool)
        order_rng = np.random.default_rng(int(branch.removeprefix('random')) if branch.startswith('random') else 0)
        objective_weights = np.ones(34) if branch == 'magnitude' else 1 + order_rng.uniform(0, 1e-7, 34)
        head = fit(active, branch, 0, None, objective_weights)
        if head is None:
            continue
        order = np.argsort(np.abs(head)) if branch == 'magnitude' else order_rng.permutation(34)
        for attempt, slot in enumerate(order[:args.max_attempts], 1):
            proposed = active.copy()
            proposed[slot] = False
            candidate = fit(proposed, branch, attempt, int(slot), objective_weights)
            if candidate is not None:
                active, head = proposed, candidate

    write_json(root / 'status.json', {'status': 'awaiting-qualification' if best and best['effective_parameter_count'] < 74 else 'incomplete',
               'run_id': root.name, 'training_mode': 'finite-domain-sparse-linear-head',
               'trial_count': len(trials), 'selection': best,
               'elapsed_seconds': round(time.monotonic() - began, 3),
               'scope': protocol['scope']})
    print(json.dumps({'run_id': root.name, 'trials': len(trials), 'best': best,
                      'elapsed_seconds': round(time.monotonic() - began, 3)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'runs/perfect-optimal-74/model.json')
    parser.add_argument('--orders', default='magnitude,random11,random23')
    parser.add_argument('--max-attempts', type=int, default=34)
    parser.add_argument('--lp-time-limit', type=float, default=5.)
    args = parser.parse_args()
    if not 0 <= args.max_attempts <= 34:
        parser.error('--max-attempts must be in [0,34]')
    main(args)
