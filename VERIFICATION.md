# Compact neural policy — current verification

The default gameplay network is `runs/perfect-sparse-49-refined/model.json`. See [SPARSE-FOLLOWUP.md](SPARSE-FOLLOWUP.md) for the further search and quantization-aware margin refinement, and [SPARSE.md](SPARSE.md) for the first sparse study.

- **49 stored learned coefficients**, 98-byte float16 payload; 24 structural zeros and one fixed unit across 74 logical slots. There are 36 hidden-layer and 13 output numeric values. JSON/structural metadata and runtime code are additional. JavaScript uses double arithmetic and eight shared geometric passes.
- Independent optimality: **4,520 / 4,520 reachable decision boards**, including forced-losing positions; worst case from empty is draw as either side.
- Actual browser inference: **zero action mismatches**, maximum raw-logit error 2.220446049250313e-16 at the unchanged 1e-9 tolerance.
- The current sparse browser worker completed **1,536 updates** in three 512-update phases at batch size 128, with pause/resume. It preserved structural zeros and fixed units and retained the certified 4,520/4,520 parent. The acceptance receipt binds the worker and inference source hashes.
- **62 ordinary tests pass**, including frozen lineage, full independent oracle fixtures, coefficient payloads, strict parity, selection history and catalog rejection checks.
- The independent **Opponent network** and **Training network** menus include 38 shared, 123 dense, **49 sparse fully optimal**, the preserved 53 and 74 fully optimal networks, and 6,537 original perfect. The 49-coefficient network is the default opponent; changing opponents starts a new game.

The preceding 53-coefficient float16 export remains selectable at `runs/perfect-sparse-53/`; it has 106-byte numeric payload, 4,520/4,520 independent optimality and zero browser action mismatches. Its completed 1,536-update worker receipt remains preserved. The new 49-coefficient model reduces both learned values and numeric payload by **7.55%**.

The first 49-coefficient export at `runs/perfect-sparse-49/` required float32. Its attempted float16 version was optimal on only **4,472/4,520** boards despite zero browser action mismatches, so it was rejected. Freezing the hidden weights at float16 and refitting the fixed readout support increased the frozen witness margin; the separate `perfect-sparse-49-refined` export then passed the full gate at float16. Both the failed attempt and successful predecessor remain intact.

The preserved 74-coefficient float32 export remains available at `runs/perfect-optimal-74/`; its independent result is also 4,520/4,520, with zero browser action mismatches and maximum error 1.4210854715202004e-14. Its earlier three-phase browser worker run completed 1,536 updates with pause/resume and retained full coverage. See [PERFECT.md](PERFECT.md).

The earlier compact choice remains available as `runs/compression-20261113-selected/model.json`. See [COMPRESSION.md](COMPRESSION.md) for its earlier no-loss search and numerical disclosure.

- **38 stored learned coefficients**, 76-byte little-endian binary16 payload; double-precision inference.
- Eight shared neural passes per move; eight learned readouts plus fixed zero; two shared geometric output biases.
- Independent worst case from empty: **draw as X and draw as O against every opponent continuation**.
- Arbitrary reachable decision boards: 3,888 / 4,520 optimal (86.02%). Full optimality is distinct from unbeatable play.
- Strict browser qualification: zero action mismatches; maximum raw-logit error 5.329070518200751e-15 under the unchanged 1e-9 tolerance.
- Its separately guarded worker completes 1,536 updates and pause/resume, preserving the certified incumbent.

Gameplay calls only frozen neural inference, fixed geometry, occupied-cell masking and the declared score rule. Teacher labels load only for explicit training or comparison. There is no solver, tactical rule or move-table fallback.

All positions are training data. Each compact result is the smallest verified found for its stated objective in the retained bounded searches, not a proof of global minimality. The failed 74-coefficient float16 export, earlier 47-coefficient parity receipt, 92-coefficient predecessor and all prior experiments remain intact. Historical full-size verification is in `docs/FULL-SIZE-VERIFICATION.md`.
