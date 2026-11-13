# Unbeatable neural student — verification, 13 November 2026

The selected 6,537-parameter policy `s11-p4` from run `20261113T233235Z-7e5be6` is optimal on every reachable decision position of ordinary 3×3 tic-tac-toe and cannot lose as either side from empty. Gameplay uses only the dense network, occupied-cell masking and lowest-index argmax ties.

| Measured property | Result |
| --- | ---: |
| Reachable boards / decision positions | 5,478 / 4,520 |
| Independently optimal neural decisions | **4,520 / 4,520** |
| Worst case as X / O, every opponent continuation | **draw / draw** |
| Parent mistakes repaired / correct decisions broken | **57 / 0** |
| Continuation updates | 4 × 3 × 512 = **6,144** |
| Browser action mismatches | **0 / 4,520** |
| Maximum independent/browser logit difference | 8.881784197001252e-15 |
| Unchanged absolute numerical tolerance | 1e-9 |
| Minimum optimal/suboptimal logit margin | 0.059817521393778605 |
| Frozen model SHA-256 | `33fd1b45032765e264dd66a3ceb239544ac20b59e77e69659f23ae6a66718072` |

Every state was deliberately supervised in this continuation. This is finite-domain qualification, not held-out generalization. The original split experiment and its imperfect outcomes remain intact; the continuation protocol records promotion of old TEST labels before optimization.

The independent verifier imports neither trainer nor teacher. It creates its own reachable census, absolute-X max/min oracle and standard-library dense inference, then explores all opponent continuations. Browser actions match every independent fixture. The runtime call-path review confirms `chooseMove(gameModel, board)` has no teacher branch or fallback. Teacher labels load only for explicit training or comparison.

The worker acceptance harness executes the identical browser module through a Node adapter. It completed all three 512×128 phases, acknowledged pause/resume, and retained the certified incumbent. Browser UI testing also exercised start, pause, stop and return to published evidence; game weights stayed fixed. Visitor training never changes the public baseline.

The ordinary suite passes **25 tests**, including six continuation artifact audits. These check full-domain admission, literal schedules, native multiplicity reconstruction, lineage, source snapshots, selection hashes, chronology, qualification receipts and exported identity, alongside original solver/API/split-study checks. JavaScript syntax checks pass. Training and model qualification used the supported CPU memory guard.

Earlier reports are preserved in `docs/INITIAL-VERIFICATION.md` and original runs. Boundaries-specific recurrence, MIR, policy-gradient arms, adapters, critic routing, multi-actor recovery and independent replay of every Adam update are not claimed.
