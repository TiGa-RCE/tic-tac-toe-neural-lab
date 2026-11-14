# Compact neural policy — current verification

See [COMPRESSION.md](COMPRESSION.md) for the search and numerical disclosure. Current model: `runs/compression-20261113-selected/model.json`.

- **38 stored learned coefficients**, 76-byte little-endian binary16 payload; double-precision inference.
- Eight shared neural passes per move; eight learned readouts plus fixed zero; two shared geometric output biases.
- Independent worst case from empty: **draw as X and draw as O against every opponent continuation**.
- Arbitrary reachable decision boards: 3,888 / 4,520 optimal (86.02%). Full optimality is distinct from unbeatable play.
- Strict browser qualification: zero action mismatches; maximum raw-logit error 5.329070518200751e-15 under the unchanged 1e-9 tolerance.
- **39 ordinary tests pass**. The guarded worker completes 1,536 updates and pause/resume, preserving the certified incumbent.
- Website training choices: 38 shared, 123 dense, and 6,537 original perfect. Gameplay keeps the 38-coefficient champion.

Gameplay calls only frozen neural inference, fixed geometry, occupied-cell masking and the declared score rule. Teacher labels load only for explicit training or comparison. There is no solver, tactical rule or move-table fallback.

All positions are training data. This is the smallest verified result found in the bounded search, not a proof of global minimality. The failed earlier 47-coefficient parity receipt and all prior experiments remain intact. Historical full-size verification is in `docs/FULL-SIZE-VERIFICATION.md`.
