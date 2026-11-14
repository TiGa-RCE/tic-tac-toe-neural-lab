# Compact neural policy — current verification

The default gameplay network is `runs/perfect-sparse-53/model.json`. See [SPARSE.md](SPARSE.md) for the follow-up search, sparse format and qualification.

- **53 stored learned coefficients**, 106-byte float16 payload; 20 structural zeros and one fixed unit across 74 logical slots. JSON/structural metadata and runtime code are additional. JavaScript uses double arithmetic and eight shared geometric passes.
- Independent optimality: **4,520 / 4,520 reachable decision boards**, including forced-losing positions; worst case from empty is draw as either side.
- Actual browser inference: **zero action mismatches**, maximum raw-logit error 2.220446049250313e-16 at the unchanged 1e-9 tolerance.
- The current sparse browser worker completed **1,536 updates** in three 512-update phases at batch size 128, with pause/resume. It preserved structural zeros and fixed units and retained the certified 4,520/4,520 parent. The acceptance receipt binds the worker and inference source hashes.
- **60 ordinary tests pass**, including frozen lineage, full independent oracle fixtures, coefficient payloads, strict parity, and catalog rejection checks.
- The independent **Opponent network** and **Training network** menus include 38 shared, 123 dense, **53 sparse fully optimal**, 74 fully optimal, and 6,537 original perfect. The 53-coefficient network is the default opponent; changing opponents starts a new game.

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
