"""Bounded fixed-zero pruning of a solver-taught, all-board-optimal policy.

Run through the supported model memory guard. Masked dense checkpoints are
search artifacts, not deployable exports: a separate sparse serialization and
independent exhaustive/browser qualification must precede publication.
"""
from train import np, Adam, atomic_json, digest, softmax
from compress_train import D4, features
from solver import enumerate_states, winner, solve, play
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import copy
import json
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parent


class Folded:
    """The existing signed-cell/D4/fixed-zero/shared-bias model, with masks."""
    def __init__(self, model):
        if (model['architecture'][0] != 9 or len(model['architecture']) != 3
                or model['architecture'][-1] != 8
                or model.get('symmetry_average') != 'D4'
                or model.get('output_bias_orbits') != 'D4-corner-zero'
                or model.get('reference_logit_zero') is not True):
            raise ValueError('Expected the existing folded signed-cell D4 architecture.')
        self.template = copy.deepcopy(model)
        self.params = [np.array(v, dtype=float) for layer in model['layers']
                       for v in (layer['weights'], layer['bias'])]
        supplied = model.get('learned_parameter_mask')
        self.masks = ([np.array(v, dtype=bool) for v in supplied] if supplied
                      else [np.ones_like(p, dtype=bool) for p in self.params])
        if any(p.shape != m.shape for p, m in zip(self.params, self.masks)):
            raise ValueError('Mask shape mismatch')
        self.project()

    def project(self):
        for p, mask in zip(self.params, self.masks):
            p[~mask] = 0.

    def clone(self):
        return Folded(self.export('clone', 'internal'))

    @property
    def count(self):
        return sum(int(mask.sum()) for mask in self.masks)

    def forward(self, x):
        n = len(x)
        blocks = []
        for perm in D4:
            transformed = np.empty_like(x)
            transformed[:, perm] = x
            blocks.append(transformed)
        all_x = np.concatenate(blocks)
        hidden = np.tanh(all_x @ self.params[0] + self.params[1])
        edge, center = self.params[3]
        bias = np.array([0., edge, 0., edge, center, edge, 0., edge])
        learned = hidden @ self.params[2] + bias
        raw = np.concatenate((learned, np.zeros((len(learned), 1))), axis=1)
        scores = sum(raw[t*n:(t+1)*n, perm] for t, perm in enumerate(D4)) / 8
        return scores, (all_x, hidden, n)

    def objective(self, x, legal, optimal, parent=None, kl=0.,
                  margin=.05, margin_weight=.5):
        z, (all_x, hidden, n) = self.forward(x)
        p = softmax(z, legal)
        mass = (p * optimal).sum(1, keepdims=True).clip(1e-30)
        loss = float(-np.log(mass).mean())
        delta = (p - p * optimal / mass) / n
        if kl:
            q = softmax(parent, legal, 2.)
            pt = softmax(z, legal, 2.)
            loss += float(kl * 4 * (q * (np.log(q.clip(1e-30))
                                      - np.log(pt.clip(1e-30)))).sum() / n)
            delta += kl * 2 * (pt - q) / n
        # A hinge on best optimal versus best suboptimal legal action directly
        # protects the declared greedy policy; all-optimal rows contribute zero.
        bad = legal & ~optimal
        good_id = np.where(optimal, z, -np.inf).argmax(1)
        bad_id = np.where(bad, z, -np.inf).argmax(1)
        rows = np.arange(n)
        gap = z[rows, good_id] - z[rows, bad_id]
        active = bad.any(1) & (gap < margin)
        loss += float(margin_weight * np.maximum(0., margin-gap)[active].sum()/n)
        ar = rows[active]
        delta[ar, good_id[active]] -= margin_weight/n
        delta[ar, bad_id[active]] += margin_weight/n
        routed = []
        for perm in D4:
            block = np.empty_like(delta)
            block[:, perm] = delta / 8
            routed.append(block[:, :8])
        raw_delta = np.concatenate(routed)
        g2 = hidden.T @ raw_delta
        g3 = np.array([raw_delta[:, [1, 3, 5, 7]].sum(), raw_delta[:, 4].sum()])
        dh = (raw_delta @ self.params[2].T) * (1-hidden**2)
        grads = [all_x.T @ dh, dh.sum(0), g2, g3]
        return loss, [g * mask for g, mask in zip(grads, self.masks)]

    def export(self, checkpoint, run):
        model = copy.deepcopy(self.template)
        model.update(checkpoint=checkpoint, run_id=run, parameter_count=self.count,
                     coefficient_precision='float64',
                     learned_parameter_mask=[m.tolist() for m in self.masks],
                     search_only_masked_dense=True,
                     deployment_requirement='Serialize omitted fixed zeros explicitly and independently qualify before play.')
        for layer, w, b in zip(model['layers'], self.params[::2], self.params[1::2]):
            layer.update(weights=w.tolist(), bias=b.tolist())
        return model


def run(args):
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    start = Path(args.start)
    parent_model = json.loads(start.read_text())
    base = Folded(parent_model)
    boards = [b for b in enumerate_states() if winner(b) is None]
    if len(boards) != 4520:
        raise ValueError('Expected exactly 4,520 decision states')
    lookup = {b: i for i, b in enumerate(boards)}
    x = features(boards, 9)
    legal = np.array([[c == 0 for c in b] for b in boards])
    optimal = np.zeros_like(legal)
    for i, b in enumerate(boards):
        optimal[i, list(solve(b)[1])] = True
    atomic_json(root/'corpus.json', {'boards': boards,
        'optimal_moves': [np.flatnonzero(row).tolist() for row in optimal],
        'scope': 'All 4,520 decision boards deliberately supervised; no held-out claim.'})
    sources = {}
    for name in ('prune_perfect.py', 'train.py', 'compress_train.py', 'solver.py'):
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
        sources[name] = digest(target)
    protocol = dict(registered_at=datetime.now(timezone.utc).isoformat(),
        start_model_sha256=digest(start), start_model=str(start),
        source_hashes=sources, corpus_sha256=digest(root/'corpus.json'),
        seeds=args.seeds, pruning_steps_per_seed=args.steps,
        candidates_per_step=args.attempts, updates_per_attempt=args.updates,
        schedule_phase_updates=512, check_every=args.check_every, batch_size=128,
        objective='Optimal-set CE throughout + best-optimal/best-suboptimal hinge + optional correct-parent T2 KL',
        target_margin=args.margin, margin_weight=args.margin_weight,
        learning_rate=args.lr, kl_coefficient=args.kl, gradient_clip=1,
        pruning='One fixed zero at a time, candidate list sorted by absolute coefficient divided by layer RMS; seed rotates among the first candidate ranks.',
        sampling='64 native/error weighted +64 uniform retention; weights1+20*native_multiplicity+1000*error; first32updates replace32native slots with current errors.',
        success='4520/4520 under 12-decimal score keys and strictly positive optimal-versus-suboptimal margin; separate independent sparse export gate required.',
        selection='Fewest retained learned coefficients, then largest minimum optimality margin, then earlier assessment.',
        scope='Bounded pruning search, not a global minimum proof. Fixed-zero masks and their structural metadata must be disclosed.')
    atomic_json(root/'pruning-protocol.json', protocol)
    state = dict(status='running', run_id=root.name,
        training_mode='finite-domain-perfect-pruning', parameter_count=base.count,
        goal='All 4,520 decisions optimal without solver fallback', trials=[], history=[],
        best=None, elapsed_seconds=0.)

    def flush():
        state['elapsed_seconds'] = round(time.monotonic()-began, 3)
        atomic_json(root/'status.json', state)

    def assess(net):
        logits = net.forward(x)[0]
        keys = np.floor(logits*1e12 + .5)
        choices = np.where(legal, keys, -np.inf).argmax(1)
        good = optimal[np.arange(len(x)), choices]
        bad = legal & ~optimal
        margin = (np.where(optimal, logits, -np.inf).max(1)
                  - np.where(bad, logits, -np.inf).max(1))[bad.any(1)]
        return {'optimal_count': int(good.sum()), 'total': len(x),
                'minimum_margin': float(margin.min()), 'choices': choices,
                'errors': np.flatnonzero(~good)}

    def summary(a):
        return {key: value for key, value in a.items() if key not in ('choices', 'errors')}

    def success(a):
        return a['optimal_count'] == 4520 and a['minimum_margin'] > 0

    def snapshot(net, optimizer, stem):
        atomic_json(root/(stem+'.json'), net.export(stem, root.name))
        arrays = {f'p{i}': p for i, p in enumerate(net.params)}
        arrays.update({f'mask{i}': m for i, m in enumerate(net.masks)})
        if optimizer is not None:
            arrays.update({f'adam_m{i}': p for i, p in enumerate(optimizer.m)})
            arrays.update({f'adam_v{i}': p for i, p in enumerate(optimizer.v)})
            arrays['adam_step'] = np.array(optimizer.t)
        np.savez_compressed(root/(stem+'.npz'), **arrays)

    initial = assess(base)
    if not success(initial):
        raise ValueError('The declared starting model must be fully optimal with positive margin')
    atomic_json(root/'parent-diagnosis.json', summary(initial))
    parent_logits = base.forward(x)[0]
    best = None
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        current = base.clone()
        for step in range(1, args.steps+1):
            # Structural zero candidates include weights and biases uniformly.
            ranked = sorted((abs(float(p[index]))/max(float(np.sqrt((p*p).mean())), 1e-12),
                             layer, index)
                            for layer, (p, mask) in enumerate(zip(current.params, current.masks))
                            for index in np.ndindex(p.shape) if mask[index])
            if not ranked:
                break
            offset = (seed % max(1, min(args.attempts, len(ranked))))
            ranked = ranked[offset:args.attempts]+ranked[:offset]
            accepted = None
            for attempt, (saliency, layer, index) in enumerate(ranked, 1):
                net = current.clone()
                net.masks[layer][index] = False
                net.project()
                optimizer = Adam(net)
                stem = f's{seed}-p{step}-a{attempt}'
                trial = {'seed': seed, 'pruning_step': step, 'attempt': attempt,
                         'parameters': net.count, 'pruned_layer': layer,
                         'pruned_index': list(index), 'saliency': saliency,
                         'phases': [], 'executed_updates': 0, 'successful': False}
                state['trials'].append(trial)
                a = assess(net)
                snapshot(net, optimizer, stem+'-u0')
                best_attempt = a['optimal_count']
                best_checkpoint = stem+'-u0'
                if not success(a):
                    for phase_start in range(0, args.updates, 512):
                        phase_size = min(512, args.updates-phase_start)
                        before = assess(net)
                        frequency = Counter()
                        trace = []
                        for start_index in rng.choice(len(boards), 128, replace=True):
                            b = boards[int(start_index)]
                            while winner(b) is None:
                                row = lookup[b]
                                move = int(before['choices'][row])
                                frequency[row] += 1
                                trace.append((row, move))
                                b = play(b, move)
                        weights = np.ones(len(boards))
                        for row, count in frequency.items():
                            weights[row] += 20*count
                        weights[before['errors']] += 1000
                        schedule = np.concatenate((
                            rng.choice(len(boards), (phase_size, 64), p=weights/weights.sum()),
                            rng.choice(len(boards), (phase_size, 64))), axis=1)
                        if len(before['errors']):
                            schedule[:min(32, phase_size), :32] = rng.choice(before['errors'], (min(32, phase_size), 32))
                        schedule_name = f'{stem}-phase{phase_start//512+1}-schedule.npz'
                        np.savez_compressed(root/schedule_name, indices=schedule,
                                            native_trace=np.array(trace, dtype=np.int64))
                        phase = {'schedule': schedule_name, 'sha256': digest(root/schedule_name),
                                 'scheduled_updates': phase_size, 'executed_updates': 0,
                                 'before': summary(before)}
                        trial['phases'].append(phase)
                        flush()  # Literal schedule and registration precede its first step.
                        for within, batch in enumerate(schedule, 1):
                            loss, grads = net.objective(x[batch], legal[batch], optimal[batch],
                                parent_logits[batch], args.kl, args.margin, args.margin_weight)
                            optimizer.step(net, grads, args.lr)
                            net.project()
                            phase['executed_updates'] = within
                            trial['executed_updates'] += 1
                            update = trial['executed_updates']
                            if update % args.check_every == 0 or within == phase_size:
                                a = assess(net)
                                row = {'seed': seed, 'pruning_step': step, 'attempt': attempt,
                                       'update': update, 'parameters': net.count, 'loss': loss,
                                       **summary(a)}
                                state['history'].append(row)
                                if a['optimal_count'] > best_attempt or success(a):
                                    best_attempt = a['optimal_count']
                                    best_checkpoint = f'{stem}-u{update}'
                                    snapshot(net, optimizer, best_checkpoint)
                                flush()
                                if success(a):
                                    break
                        snapshot(net, optimizer, f'{stem}-phase{phase_start//512+1}-end')
                        if success(a):
                            break
                trial.update(successful=success(a), best_optimal_count=best_attempt,
                             best_checkpoint=best_checkpoint, final=summary(a))
                if success(a):
                    accepted = net
                    checkpoint = f'{stem}-accepted'
                    snapshot(net, optimizer, checkpoint)
                    candidate = {'checkpoint': checkpoint, 'parameters': net.count,
                                 'seed': seed, 'pruning_step': step, **summary(a)}
                    if (best is None or candidate['parameters'] < best['parameters']
                            or (candidate['parameters'] == best['parameters']
                                and candidate['minimum_margin'] > best['minimum_margin'])):
                        best = candidate
                        atomic_json(root/'model.json', net.export(checkpoint, root.name))
                        atomic_json(root/'selection-before-qualification.json', {'selected': best,
                            'source_protocol_sha256': digest(root/'pruning-protocol.json'),
                            'model_sha256': digest(root/'model.json'),
                            'qualification': 'Training diagnosis only; independent sparse export qualification required.'})
                        state['best'] = best
                    flush()
                    break
                flush()
            if accepted is None:
                break
            current = accepted
    state['status'] = 'awaiting-qualification' if best else 'incomplete'
    flush()
    atomic_json(root/'search-trials.json', state['trials'])
    print(json.dumps({'run_id': root.name, 'status': state['status'], 'best': best,
                      'trial_count': len(state['trials']), 'elapsed_seconds': state['elapsed_seconds']}))
    return 0 if best else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--start', default=str(ROOT/'runs/perfect-optimal-74/model.json'))
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--updates', type=int, default=1024)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--seeds', default='11,23,37')
    p.add_argument('--check-every', type=int, default=64)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--kl', type=float, default=.001)
    p.add_argument('--margin', type=float, default=.05)
    p.add_argument('--margin-weight', type=float, default=.5)
    a = p.parse_args()
    a.seeds = [int(value) for value in a.seeds.split(',')]
    if not (1 <= a.steps <= 73 and 1 <= a.updates <= 16384
            and 1 <= a.attempts <= 20 and 1 <= a.check_every <= a.updates
            and a.seeds and all(0 <= seed < 2**32 for seed in a.seeds)
            and 0 < a.lr <= .1 and 0 <= a.kl <= 1
            and 0 < a.margin <= 10 and 0 <= a.margin_weight <= 10):
        p.error('Expected bounded positive search settings and unsigned 32-bit seeds.')
    try:
        return run(a)
    except Exception:
        root = Path(a.run_dir)
        if root.exists():
            (root/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
