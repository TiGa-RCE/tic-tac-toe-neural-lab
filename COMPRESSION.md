# Network compression — 13 November 2026

The published game now uses **38 learned coefficients**, down from 6,537: **99.4187% fewer**, or about 172 times fewer. They occupy **76 bytes** as IEEE binary16 values. JSON metadata, geometric maps and runtime code are additional. Inference uses JavaScript double precision.

| Network | Coefficients | Neural passes per move | Optimal arbitrary boards | Worst-case X / O |
| --- | ---: | ---: | ---: | --- |
| Original perfect student | 6,537 | 1 | 4,520 / 4,520 | draw / draw |
| Compact dense 9→6→9 | 123 | 1 | 3,800 / 4,520 | draw / draw |
| Selected shared 9→2→8 plus fixed zero | **38** | **8** | **3,888 / 4,520** | **draw / draw** |

All three appear in the website's **Training network** menu. The chosen model supplies the visitor worker's initial weights and architecture. Training a different model never replaces the compact game baseline. Run downloads contain the parent, phase tensors, selected result and literal schedules.

## Search boundary

The raw search recorded 59 seeded trials across 21 executed architecture families: signed 9-cell, two-channel 18-input or one-hot 27-input encodings; one or two hidden layers; and affine baselines. The D4 search recorded 12 trials across five families. Seeds 11, 23 and 37 received up to eight or twelve phases, respectively, with 512 Adam updates per phase. Larger networks were skipped once smaller successes existed.

The smallest raw success had 123 parameters. A two-neuron D4 network used 47 before folding. Smaller one-neuron and narrow bottleneck families did not reach no-loss within their registered budgets. These failures are preserved, but **do not prove impossibility**. This is the smallest verified result found here, not a proven global minimum.

Training retains native collection, targeted review and balanced replay. Current-network actions generate the traces. Exact adversarial diagnosis identifies unsafe states reachable against all opponent continuations. Half each batch uses weighted native/adversarial replay; half uses uniform retention. The first 32 updates per phase target unsafe states at a 25% share. First-phase optimal-set CE is followed by solver-labeled safe-action-set CE; already-losing positions accept every legal action to focus capacity on avoiding losses. Retention KL uses temperature 2 and coefficient 0.01. Adam uses 0.01 for the first two phases, 0.003 later, and norm clipping at one. All positions are teaching data, with no held-out claim.

## Algebraic format and numerical policy

The same learned network scores all eight square rotations/reflections. Its logits map back to the original cells and average. This shares weights at a cost of eight neural passes. Geometry uses no board-specific tactics, solver call or move lookup.

Subtracting the ninth readout's weights and bias from the others permits eight learned scores plus a fixed ninth zero. D4 averaging makes output biases depend only on corner, edge or center position. Removing the common corner offset leaves two learned biases. The count is 18 first-layer weights + 2 hidden biases + 16 readout weights + 2 shared biases = **38**.

The original 47-coefficient D4 export failed strict browser parity on 57 numerical ties, despite only 3.55e-15 raw-logit differences. Its failed receipt is unchanged. The new policy explicitly ranks `floor(averaged_logit*10^12+0.5)`, then lowest legal cell index. The fixed 1e-9 numerical tolerance and exact action agreement requirements were not loosened. The folded, float16-valued policy was independently requalified and has zero action mismatches.

`coefficients.bin` is little-endian IEEE binary16: each layer's row-major weights followed by its stored bias array, giving 18 weights, 2 biases, 16 weights and 2 biases. It is 76 bytes. `model.json` supplies the complete portable model used by the game.

## Fresh qualification

- Independent oracle and inference: 3,888 of all 4,520 decision positions are optimal.
- Every opponent continuation from empty: worst-case draw as X and draw as O, with no loss.
- Browser parity: zero action mismatches; maximum logit error 5.329070518200751e-15, tolerance 1e-9.
- Actual browser worker: 1,536 Adam updates, three literal 512×128 phases, pause/resume acknowledged, certified incumbent retained.
- Live selector: the 6,537-parameter training choice uses that architecture while gameplay retains the 38-coefficient model; pause, stop and return to published evidence work.
- Ordinary verification: 39 tests pass, including lineage, hashes, payload, format and preserved-failure checks.

Evidence is under `runs/compression-20261113-a/`, `runs/compression-20261113-d4/` and `runs/compression-20261113-selected/`. Every older run remains preserved. A separately timestamped post-qualification receipt binds the source snapshots without implying earlier source-hash registration.
