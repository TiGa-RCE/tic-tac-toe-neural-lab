# Compact neural policy — current verification

The added fully optimal choice is `runs/perfect-optimal-74/model.json`. See [PERFECT.md](PERFECT.md) for the 54-trial search and precision results.

- **74 stored learned coefficients**, 296-byte float32 payload; JavaScript double arithmetic and eight shared geometric passes.
- Independent optimality: **4,520 / 4,520 reachable decision boards**, including forced-losing positions; worst case from empty is draw as either side.
- Actual browser inference: **zero action mismatches**, maximum raw-logit error 1.4210854715202004e-14 at the unchanged 1e-9 tolerance.
- The identical browser training worker completed **1,536 updates**, with pause/resume and 4,520/4,520 retained. Its frozen certified parent remains selected.
- **48 ordinary tests pass**, including frozen lineage, full independent oracle fixtures, coefficient payloads, strict parity, and catalog rejection checks.
- The public Training network menu includes 38 shared, 123 dense, **74 fully optimal**, and 6,537 original perfect.

The playable game retains `runs/compression-20261113-selected/model.json`. See [COMPRESSION.md](COMPRESSION.md) for its earlier no-loss search and numerical disclosure.

- **38 stored learned coefficients**, 76-byte little-endian binary16 payload; double-precision inference.
- Eight shared neural passes per move; eight learned readouts plus fixed zero; two shared geometric output biases.
- Independent worst case from empty: **draw as X and draw as O against every opponent continuation**.
- Arbitrary reachable decision boards: 3,888 / 4,520 optimal (86.02%). Full optimality is distinct from unbeatable play.
- Strict browser qualification: zero action mismatches; maximum raw-logit error 5.329070518200751e-15 under the unchanged 1e-9 tolerance.
- Its separately guarded worker completes 1,536 updates and pause/resume, preserving the certified incumbent.

Gameplay calls only frozen neural inference, fixed geometry, occupied-cell masking and the declared score rule. Teacher labels load only for explicit training or comparison. There is no solver, tactical rule or move-table fallback.

All positions are training data. Each compact result is the smallest verified found for its stated objective in the retained bounded searches, not a proof of global minimality. The failed 74-coefficient float16 export, earlier 47-coefficient parity receipt, 92-coefficient predecessor and all prior experiments remain intact. Historical full-size verification is in `docs/FULL-SIZE-VERIFICATION.md`.
