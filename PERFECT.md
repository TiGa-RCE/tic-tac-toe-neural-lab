# Compact policy optimal on every decision board

The selected network has **74 learned float32 coefficients**, with a **296-byte numeric payload**, and chooses an optimal action on **all 4,520 reachable nonterminal tic-tac-toe boards**. Independent exhaustive qualification also confirms a worst-case draw from empty as both X and O. The browser implementation selects exactly the same actions on all 4,520 fixtures, with maximum absolute logit error `1.4210854715202004e-14` at the unchanged `1e-9` tolerance.

This is the **smallest fully optimal network verified in this retained, bounded search**. It is not a global minimum proof. The 74-coefficient model is the default opponent. The **Opponent network** menu offers all four certified networks; changing that choice starts a new game. The earlier 38-coefficient network remains available: it is unbeatable from empty but optimal on 3,888/4,520 arbitrary decision boards. The independent **Training network** menu also offers the **Compact fully optimal network** for browser experiments. The original 6,537-parameter perfect network and the intermediate 92-coefficient perfect export remain preserved.

## Network and gameplay

The model takes nine signed cell values relative to the next player and has four tanh hidden neurons. Its eight learned linear readouts are accompanied by a fixed ninth score of zero. The coefficient count is `9×4 + 4 + 4×8 + 2 = 74`: first-layer weights, hidden biases, readout weights, and two shared geometric output biases.

One shared network evaluates all eight D4 rotations/reflections. Scores are mapped back to the original cells and averaged. This fixed geometric averaging requires eight forward passes, using one set of weights rather than separately trained ensemble members. The export removes the common ninth-output score and folds output biases into D4 corner/edge/center orbits, with the common corner bias fixed to zero. These algebraic operations remove redundant coefficients without adding a move table or tactical features.

Gameplay uses neural scores, occupied-cell masking, scores rounded to 12 decimal places, and the declared greedy tie rule. There is **no solver fallback**. A deterministic solver supplies teaching labels and independent qualification; optional teacher comparison is separate from move selection. Float32 describes the stored coefficient values; the browser computes with ordinary JavaScript double arithmetic. The 296-byte count covers only packed little-endian coefficients, excluding JSON metadata, fixed geometry and runtime code.

## Training recipe and finite-domain scope

All 4,520 decision boards are deliberately supervised, including forced-losing positions. Each target contains every equally optimal move according to exact minimax value. The loss is the negative logarithm of the total probability assigned to that optimal-action set. Safe-only targets are never substituted. This measures complete finite-domain optimality, with no held-out generalization claim.

Every phase first records actual neural actions in native play. The next literal schedule is saved before execution. Each 128-example batch contains 64 weighted native/error samples and 64 uniform retention samples. Sampling weights are `1 + 20 × native multiplicity + 1000 × current optimality-error indicator`; during the first 32 updates, current errors replace 32 native slots when present. Native traces remain honest records of neural actions: teacher labels do not override those actions.

The broad searches use Adam, global gradient norm clipping at 1, and weak `0.001 × T² KL(parent || candidate)` retention at `T=2`, restricted to positions where the frozen parent is correct. Base learning rates are 0.01 in phases 1–2, 0.003 in phases 3–6 and 0.001 thereafter. The focused warm restarts use one tenth of those rates, no KL, and fresh Adam initialized from saved weights. They retain optimal-set supervision and the same native/error sampling and uniform retention.

Trials stop successfully only at 4,520/4,520 under the deployed rounded-score rule. Unexecuted rows in an already saved schedule remain distinguishable from executed updates. Protocol/corpus hashes, source snapshots, literal schedules, native traces, tensors, optimizer states and failed trials are retained alongside the successful checkpoints.

## Retained search

The all-board-optimal study ran **54 trials**, separately from the earlier 71-trial no-loss compression study.

| Search | Executed trials | Budget and method | Result |
| --- | ---: | --- | --- |
| A | 32 | Signed-cell encoding, one/two hidden layers, raw and D4 policies; seeds 11/23/37; up to 12×512 updates, checks every 64 | Smallest successful export: 92 coefficients, five hidden neurons |
| B | 18 | D4 two-channel/one-hot encodings, 1–3 hidden neurons; seeds 11/23/37; 16×512 updates, checks every 64 | No full pass; best 4,472/4,520, planned 83 coefficients |
| C | 1 | Four-neuron signed-cell D4 warm restart; up to 32×512 updates, checks every 16; 0.1 learning-rate scale, KL 0 | Full pass at seed 11, phase 9, update 16; selected 74-coefficient export |
| D | 3 | Three-neuron signed-cell D4 warm restart; seeds 11/23/37; 32×512 updates, checks every 16; 0.1 learning-rate scale, KL 0 | No full pass; best 4,474/4,520, planned 56 coefficients |

Search C resumed the preserved A checkpoint `a9-4-9-d4-s11-p9-u64.json` and needed 4,112 additional updates. Search D's smaller failed candidates establish the result of those bounded attempts, not impossibility at that size. The registered families and exact budgets are in each search's `perfect-protocol.json`. A and C retain `awaiting-qualification` as their training-only status; the separate finalized export receipts record completed independent qualification.

## Export and independent evidence

The selected training checkpoint was frozen before folding and precision qualification. Float16 was tested first: it retained numerical browser parity but reached only 4,516/4,520 and failed no-loss qualification. That attempt is preserved. Float32 passed all acceptance conditions and was selected; numerical parity alone never establishes policy optimality.

| Evidence | Location |
| --- | --- |
| Final selection and minimality scope | [`runs/perfect-optimal-selection.json`](runs/perfect-optimal-selection.json) |
| Selected coefficients and metadata | [`runs/perfect-optimal-74/model.json`](runs/perfect-optimal-74/model.json) |
| Exhaustive coverage and browser parity | [`runs/perfect-optimal-74/evaluation.json`](runs/perfect-optimal-74/evaluation.json) |
| Precision attempts, including failed float16 | [`runs/perfect-optimal-74/transformation-attempts.json`](runs/perfect-optimal-74/transformation-attempts.json) |
| Pre-qualification transform/source binding | [`runs/perfect-optimal-74/transformation-before-qualification.json`](runs/perfect-optimal-74/transformation-before-qualification.json) |
| Qualification consumption hashes | [`runs/perfect-optimal-74/qualification-consumption.json`](runs/perfect-optimal-74/qualification-consumption.json) |
| Browser-worker training acceptance | [`runs/perfect-optimal-74/browser-worker-verification.json`](runs/perfect-optimal-74/browser-worker-verification.json) |
| Earlier full-optimal export | [`runs/perfect-optimal-92/`](runs/perfect-optimal-92/) |
| Search evidence | `runs/perfect-optimal-search-a/` through `runs/perfect-optimal-search-d/` |

The selected `model.json` SHA-256 is `f2126224b52abde726908257a1faa918bf84247601f799f31876c64e2443f9f7`. Independent qualification uses an absolute-X oracle and explores every possible opponent continuation from empty for both sides. The portable audit compares exact chosen actions as well as logits on every decision position.

The worker acceptance run executed all 1,536 updates in three 512-update phases at batch size 128, acknowledged pause/resume, and retained the certified 4,520/4,520 training parent. Visitor experiments remain isolated from the published gameplay checkpoint. A dashboard preview is saved at [`docs/perfect-network-preview.jpg`](docs/perfect-network-preview.jpg).

## Reproduction

Use Python 3.10+, NumPy and Node 18+ from the repository root. Preserve original run directories; the commands below create new names. On this workstation, training and deliberate model qualification use the supported memory guard. Before allocating a new search run, admit a conservative 1 GB of growth in the existing external workspace:

```bash
python3 ~/.codex/bin/storage_admit.py --path . --expected-bytes 1000000000
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 perfect_search.py --run-dir runs/reproduce-perfect-a --phases 12 --updates 512 --check-every 64 --seeds 11,23,37 --kl 0.001
```

The focused successful search can be reproduced from its retained, unfolded warm checkpoint. Recheck storage admission before this additional growth phase:

```bash
python3 ~/.codex/bin/storage_admit.py --path . --expected-bytes 1000000000
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 perfect_search.py --run-dir runs/reproduce-perfect-c --architectures '[[9,4,9]]' --phases 32 --updates 512 --check-every 16 --seeds 11,23,37 --kl 0 --lr-scale 0.1 --resume runs/perfect-optimal-search-a/a9-4-9-d4-s11-p9-u64.json
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 finalize_perfect.py --source-run runs/reproduce-perfect-c --run-dir runs/reproduce-perfect-export
```

For an independent recheck of the published export, without retraining:

```bash
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 verify_export.py --model runs/perfect-optimal-74/model.json --output perfect-recheck.json
```

Ordinary metadata/unit checks and the static build are:

```bash
python3 -m unittest discover -s tests -v
python3 build_site.py --run-dir runs/perfect-optimal-74 --output site/dist
```

The builder publishes the independently qualified 74-coefficient network as the default game and keeps all four certified choices in the catalog. See [README.md](README.md) for local serving and browser training, and [COMPRESSION.md](COMPRESSION.md) for the earlier no-loss search.
