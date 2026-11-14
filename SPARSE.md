# Further compression with explicit sparse coefficients

This follow-up searches below the preserved 74-coefficient network while keeping the same acceptance requirement: every one of the 4,520 reachable decision boards must receive an optimal neural action, and the exact deployed browser implementation must agree with independent inference. All boards are deliberate teaching data; this is complete finite-domain coverage, with no held-out generalization claim or global minimum proof.

The selected network has **53 learned float16 coefficients**, a **106-byte numeric payload**, and four tanh hidden neurons. It is independently optimal on **4,520/4,520 boards**, with a worst-case draw as X and O. Actual browser inference has **zero action mismatches** and maximum raw-logit error `2.220446049250313e-16` under the unchanged `1e-9` tolerance. It is the new default opponent; the 74-coefficient network and every earlier choice remain available.

The selected export is named by [`runs/sparse-perfect-selection.json`](runs/sparse-perfect-selection.json). The original [74-coefficient study](PERFECT.md), its frozen model, and all previous network choices remain available.

## What the coefficient count means

The `sparse-fixed-v1` format stores a numeric value for each retained learned coefficient, `null` for an exact structural zero, and an explicit `fixed:+1` or `fixed:-1` marker for a fixed unit coefficient. Every numeric weight and bias counts. The position of each zero or unit is structural metadata; JSON, those positions, fixed geometry and runtime code are additional storage. The packed coefficient payload excludes that overhead.

The final 53-coefficient export stores **39 hidden-layer coefficients and 14 output coefficients**, with **20 structural zeros and one fixed `-1`** across the original 74 logical slots. The coefficient reduction is **28.38%** from 74; the numeric payload reduction is **64.19%** from the previous 296-byte float32 payload. This result changes coefficient count and precision, while retaining four hidden neurons and eight forward passes.

One output coefficient can be fixed to unit magnitude by dividing every learned output weight and output bias by the same positive constant. Positive common scaling preserves score order. The discarded scale is absent from the deployed model; this removes one redundant learned scalar, rather than removing another neuron. Independent qualification still checks the final quantized values under the declared 12-decimal score rule, because floating-point rounding can alter a policy.

The sparse network remains an ordinary tanh policy evaluated over the eight fixed D4 rotations/reflections. Mapped neural logits are averaged, occupied cells are masked, and the highest legal score determines the move. Structural zeros and fixed units do not select actions from board-dependent rules. There is no tactical rule, position table or solver fallback during play. Policy probabilities can change under output normalization; they are not win probabilities.

## Search methods

Three bounded approaches extend the same teacher, native collection, review and retention recipe:

- **Pruning and fine tuning:** remove an explicit coefficient, preserve its zero mask, and optimize the remaining weights using optimal-action-set cross entropy plus a ranking margin. Actual native actions, targeted current errors, uniform retention, literal schedules, tensors and Adam state are recorded. A step is accepted only after all 4,520 rounded greedy actions are optimal and the optimal-versus-suboptimal margin is positive.
- **Smaller hidden layer polishing:** use L-BFGS on a three-neuron network with optimal-set cross entropy and annealed smooth ranking margins. Each phase records actual native traces and frozen full-domain review/retention weights before optimization. The gradient is checked by central finite differences inside the guarded run. The first successful training checkpoint has 56 coefficients after the existing common-logit and D4 bias folds; its training result is separate from export qualification.
- **Sparse linear readout fitting:** freeze the 74-coefficient model's 40 hidden-layer coefficients and fit its output head with linear constraints. A frozen witness from the exact optimal-action set must outscore every suboptimal legal action. The registered constraint matrix contains 7,304 rows. Sequential zero constraints are committed only when the linear program is feasible and every greedy neural decision remains optimal. The linear program and deterministic game teacher run only during training; neither is called by gameplay.

The first qualified sparse export was [`runs/perfect-sparse-54/`](runs/perfect-sparse-54/): 40 hidden coefficients, 14 learned output coefficients, 19 structural zeros and one fixed unit across the original 74 slots. Its 54 float16 coefficients pack into 108 bytes. Independent and browser qualification both grade all 4,520 decisions optimal, with zero browser action mismatches and maximum raw-logit error `3.3306690738754696e-16` at the unchanged `1e-9` tolerance. This intermediate evidence remains preserved.

The follow-up pruned one further hidden-layer coefficient and refit the remaining values, producing a 54-active-coefficient training checkpoint. Positive output normalization then removed one redundant learned scale. The resulting [`runs/perfect-sparse-53/`](runs/perfect-sparse-53/) export passed the complete independent and browser gates at float16 and supersedes the preserved 54-coefficient intermediate.

| Bounded search | Executed attempts | Result before final sparse normalization |
| --- | ---: | --- |
| Pruning A | 46 pruning/fine-tuning attempts | 64 active coefficients |
| Sparse readout fit | 105 linear programs | 55 active coefficients; 54 after fixed-unit normalization |
| Polishing E | 2 starting networks | Full-domain success with 56 planned folded coefficients and three hidden neurons |
| Pruning B | 28 pruning/fine-tuning attempts | 54 active coefficients; selected 53 after fixed-unit normalization |
| Pruning C | 18 pruning/fine-tuning attempts from the three-neuron candidate | No smaller full-optimal result in its budget |

## Evidence and reproduction

Search receipts are retained in `runs/perfect-pruning-a/`, `runs/perfect-pruning-b/`, `runs/perfect-pruning-c/`, `runs/perfect-optimal-polish-e/`, and `runs/perfect-sparse-head-a/`. Protocols bind source snapshots, teaching corpus, native traces and frozen schedules or constraints. Final export receipts bind the source model, sparse transformation, precision attempts, independent full-domain fixtures and exact browser parity. Failed smaller attempts remain part of the bounded evidence.

The selected model SHA-256 is `22af9dcaabff1439a0c11723fc7eb32e1767989789c58e2ae053960fc2a5301c`. Its complete independent fixtures, portable parity, precision attempts and qualification consumption hashes are retained in [`runs/perfect-sparse-53/`](runs/perfect-sparse-53/). The selection registry binds this export without replacing the original dense perfect-model registry.

The identical browser training worker completed all **1,536 updates**, three 512-update phases at batch size 128, with pause/resume acknowledged. It retained 4,520/4,520 optimal coverage and the certified parent while preserving every structural zero and fixed unit. Its [`browser-worker-verification.json`](runs/perfect-sparse-53/browser-worker-verification.json) receipt binds both worker and inference source hashes. **60 ordinary tests pass**, including independent frozen evidence, sparse serialization, coefficient accounting, selection isolation and catalog rejection checks.

Sparse finalization tests float16, float32 and float64 in registered order, selecting the first precision that passes the complete unchanged gate. Browser parity by itself is insufficient: independent optimality and no-loss qualification as both X and O must pass too. The public builder rejects an unqualified sparse choice before duplicate-model filtering.

Run ordinary metadata and interface checks without a model backend:

```bash
python3 -m unittest discover -s tests -v
python3 build_site.py --run-dir runs/perfect-sparse-53 --output site/dist
```

On this workstation, admit storage before a new search and use the supported memory guard for deliberate training, inference or export qualification. Install `requirements-research.txt` for SciPy-based L-BFGS and linear programming; the original Adam trainers require NumPy. For example, preserve existing evidence and use a new run name:

```bash
python3 -m pip install -r requirements-research.txt
python3 ~/.codex/bin/storage_admit.py --path . --expected-bytes 500000000
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 polish_perfect.py --run-dir runs/reproduce-polishing --iterations 150 --phases 6
```

The public site's **Opponent network** and **Training network** selectors stay independent. Visitor training never substitutes experimental weights for the frozen opponent. See [README.md](README.md) for local serving and [VERIFICATION.md](VERIFICATION.md) for the current selected export and completed checks.
