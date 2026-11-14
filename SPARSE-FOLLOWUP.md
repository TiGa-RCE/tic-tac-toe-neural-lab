# From 53 to 49 fully optimal learned coefficients

The new default is [`runs/perfect-sparse-49-refined/model.json`](runs/perfect-sparse-49-refined/model.json): **49 learned float16 values in 98 bytes**, independently optimal on **all 4,520 reachable decision boards**. Both worst-case results from empty are draws, as X and O. Gameplay still consists only of neural inference, fixed D4 geometry, occupied-cell masking and the declared greedy score rule. No optimizer, teacher, position table or solver fallback chooses a move during play.

The reduction from the preserved 53-coefficient model is **7.55%** in learned values and numeric payload. The network retains four tanh hidden neurons and eight shared geometric passes. Its 74 logical slots contain **36 hidden-layer numeric values, 13 output numeric values, 24 structural zeros and one fixed unit**. Positions, fixed geometry, JSON and runtime code are additional storage. The `sparse-fixed-v1` format and positive-output-scale fold are described in [SPARSE.md](SPARSE.md).

The **Opponent network** and **Training network** menus retain the 53-coefficient model as an earlier fully optimal choice, together with the 38-, 74-, 123- and 6,537-coefficient models. Their selections remain independent. Changing opponents starts a new game with frozen weights; visitor experiments never replace the selected opponent.

## Bounded search and retained failures

All 4,520 decision boards remain deliberate teaching data, including forced-losing positions. Native traces record actual neural choices. Review targets and full-domain retention data are frozen before fitting. This measures finite-domain coverage, with no held-out generalization claim.

| Search | Recorded attempts | Observed result |
| --- | ---: | --- |
| Greedy readout fitting B | 105 linear programs | Did not improve the 53-coefficient champion |
| Adam pruning D | 60 pruning/fitting attempts | No full-optimal smaller result; best 4,504/4,520 |
| Masked L-BFGS F | 28 candidate masks | 51 active numeric slots; separate normalization/export qualified 50 learned float16 values, 100 bytes |
| Iterative readout C | 260 LP/MILP solver attempts | 14 active readout slots with 36 hidden coefficients; separate normalization/export qualified 49 learned float32 values |
| Masked L-BFGS G | 6 candidate masks | No smaller full-optimal result; best 4,516/4,520 |
| Margin refinement with quantized hidden weights | One fixed-support linear program | Final 49 learned float16 values, 98 bytes, all 4,520 optimal |

Masked L-BFGS fits only unmasked coordinates and keeps structural zeros fixed. It combines optimal-action-set cross entropy with an annealed smooth ranking margin, and retains a beam of distinct successful masks. Each phase saves the native trace and deterministic full-domain review/retention weights before optimization. A central finite-difference check validates its free-coordinate gradient inside the guarded run. Failed best checkpoints and optimizer endpoints remain available.

The iterative readout search freezes the hidden features and refreshes neural witnesses between bounded rounds. Every witness is an actual neural action already graded optimal by the exact teacher. LP/MILP constraints require it to outscore every suboptimal legal action. The mixed-integer search reports 14 active head slots for its registered problem: fixed hidden features and witnesses, coefficient bound **M=1**, margin **tau=1e-5**, and a selected unit anchor of **+1 or -1**. That result is conditional on those features, witnesses, bounds, margin and anchor. It is neither a global minimum proof for neural networks nor a proof that every alternative support or hidden representation needs that many coefficients.

## Why the final float16 refinement was needed

The first 49-coefficient export is retained in [`runs/perfect-sparse-49/`](runs/perfect-sparse-49/). Direct float16 rounding reached only **4,472/4,520**, even though browser actions matched independent inference exactly. It therefore failed the full-optimality gate. Float32 passed, producing a preserved 196-byte learned-value payload.

The refinement first freezes the **36 hidden coefficients at their float16 values**, then refits the same **14-slot output support**, including the fixed unit anchor. A bounded linear program maximizes the smallest frozen witness-versus-suboptimal margin. Its fitted margin was `0.00013154206031162552`; the actual measured minimum gap was `0.0001315420603062445`. This improves tolerance to output quantization while retaining the registered support and unit bound. Final sparse serialization removes the redundant output-scale scalar and independently tests the quantized result.

The new [`runs/perfect-sparse-49-refined/`](runs/perfect-sparse-49-refined/) export passes at float16. The original failed float16 attempt, float32 predecessor, 50-coefficient intermediate and earlier 53-coefficient champion remain unchanged. Exact browser parity alone never substitutes for complete optimality and both-side no-loss qualification.

## Independent qualification and browser training

- **4,520/4,520 independent optimal actions**, with no mistakes and worst-case draws as both sides.
- **Zero browser action mismatches**; maximum raw-logit error `2.220446049250313e-16` at the unchanged absolute tolerance `1e-9`.
- **49 numeric learned values**, 98-byte float16 payload, 24 structural zeros and one explicit fixed unit; all weight and bias values count.
- The identical browser training worker completed **1,536 updates**, three phases of 512 at batch size 128, acknowledged pause/resume, and retained the certified 4,520/4,520 parent. Every structural zero and fixed unit remained fixed.
- **62 ordinary tests pass**, covering frozen evidence, coefficient and precision accounting, strict qualification, selector isolation, and historical catalog rejection gates.

The selected model SHA-256 is `2f2eed519d197bb498cb80127d5ca22ff66257b635db0eaa426997c465bf1c2d`. Its independent fixtures, precision attempts, portable verification, source snapshots and qualification-consumption hashes are in the final run. The current registry is [`runs/sparse-perfect-selection.json`](runs/sparse-perfect-selection.json), with the previous 53-coefficient run retained in `previous_run_ids`.

| Evidence | Location |
| --- | --- |
| Final qualification and numerical results | [`runs/perfect-sparse-49-refined/evaluation.json`](runs/perfect-sparse-49-refined/evaluation.json) |
| Final transformation and source bindings | [`runs/perfect-sparse-49-refined/transformation-before-qualification.json`](runs/perfect-sparse-49-refined/transformation-before-qualification.json) |
| Browser training acceptance | [`runs/perfect-sparse-49-refined/browser-worker-verification.json`](runs/perfect-sparse-49-refined/browser-worker-verification.json) |
| Quantized-hidden margin protocol | [`runs/perfect-head-margin-f16/refine-head-margin-protocol.json`](runs/perfect-head-margin-f16/refine-head-margin-protocol.json) |
| First 49-coefficient precision failure and success | [`runs/perfect-sparse-49/transformation-attempts.json`](runs/perfect-sparse-49/transformation-attempts.json) |
| Masked fitting search | [`runs/perfect-masked-polish-f/`](runs/perfect-masked-polish-f/) |
| Iterative LP/MILP search | [`runs/perfect-iterative-head-c/`](runs/perfect-iterative-head-c/) |
| Previous fully optimal champion | [`runs/perfect-sparse-53/`](runs/perfect-sparse-53/) |

## Reproduction

Use `requirements-research.txt` for SciPy-based research tools. Preserve all existing run directories. On this workstation, admit storage before new search allocations and use the supported memory guard for deliberate model training, conversion, inference and qualification.

```bash
python3 -m pip install -r requirements-research.txt
python3 ~/.codex/bin/storage_admit.py --path . --expected-bytes 500000000
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 masked_polish.py --run-dir runs/reproduce-masked --steps 3 --attempts 6 --iterations 150 --phases 4 --beam-width 2 --seeds 11
```

To reproduce the last fixed-support margin refinement from the retained iterative model, recheck storage admission before the new growth phase:

```bash
python3 ~/.codex/bin/storage_admit.py --path . --expected-bytes 500000000
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 refine_head_margin.py --run-dir runs/reproduce-margin-f16 --model runs/perfect-iterative-head-c/model.json --hidden-precision float16
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 finalize_sparse.py --source runs/reproduce-margin-f16/model.json --run-dir runs/reproduce-49-export --normalize
```

Ordinary checks and the current public static build are:

```bash
python3 -m unittest discover -s tests -v
python3 build_site.py --run-dir runs/perfect-sparse-49-refined --output site/dist
```

These searches establish the smallest verified result retained here. Failed smaller attempts and the conditional mixed-integer result do not establish a global network minimum.
