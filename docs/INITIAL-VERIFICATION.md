# Verification — 13 November 2026

The local game/dashboard is implemented, the solver-trained student has completed training, and the selected export has passed independent inference and action-parity checks. The network is imperfect; no unbeatable-policy claim is made.

Verified run: `20261113T232051Z-9ad6fe`, master seed 42, selected checkpoint `s37-p3`, 6,537 parameters.

| Frozen measurement | Result |
| --- | ---: |
| TRAIN decisions | 3,147 / 3,147 optimal |
| Validation decisions | 439 / 443 optimal (99.10%) |
| META finalist decisions | 466 / 480 optimal (97.08%) |
| TEST decisions | 411 / 450 optimal (91.33%) |
| Exhaustive reachable decisions | 4,463 / 4,520 optimal (98.74%) |
| Exhaustive mistakes | 57 |
| Selected validation repairs / breaks vs founder | 4 / 1 |
| Worst-case result from empty as X | draw |
| Worst-case result from empty as O | loss |
| Independent/browser action mismatches | 0 / 4,520 |
| Maximum independent/browser logit difference | 7.105427357601002e-15 |
| Fixed absolute logit tolerance | 1e-9 |
| Finite-difference objective/gradient max error | 8.4e-12 |

The exhaustive figure includes TRAIN. The fixed finite partition was also used in the preserved smoke run, so the final TEST score is a repeated-partition diagnostic rather than a fresh independent research campaign. No architecture, optimizer hyperparameter, seed, or selection rule was changed in response to TEST scores. Review tightened input boundaries and added audit receipts; original smoke evidence/source bytes remain intact.

Fresh verification:

- `python3 -m unittest discover -s tests -v`: 19 tests pass. Covers exact teacher values/census, invalid moves, D4 partitions, TRAIN-only literal schedules/native traces, source/model/receipt hashes and chronology, loopback/same-origin API checks, guarded launch/storage refusal, and control messages. Tests load no learned tensors.
- `node --check` for the two browser modules and portable verifier passes.
- Training and deliberate model qualification ran through the local `memory-run` launcher with a 512 MiB declared peak; storage admission passed. No package install or GPU/model download was needed.
- An independently implemented standard-library oracle and dense forward pass graded all 4,520 decisions and enumerated every adversarial opponent continuation from empty. Exported browser inference matched all chosen actions and logits within the fixed tolerance. The oracle imports neither trainer nor teacher.
- Live browser verification: start → pause at an update boundary → resume → completed; ongoing game retained its earlier phase checkpoint while training advanced; new game loaded the final checkpoint; both X/O roles and teacher comparison worked; a complete neural win correctly locked the board.
- A separate guarded worker with a cooperative stop request ended `cancelled`; the final completed run remained selected.
- Final browser layout had content width equal to viewport width at 659 px, with no horizontal overflow. Captured browser error log was empty. Tiny audit values use scientific notation.

The web server remains available on `127.0.0.1:8765` for this local dashboard. No training job remains active. The project was not published to a remote host; its trainer uses this Mac's local guarded Python process.

Known limits: Sudoku source was not available; the recipe is grounded in the supplied Boundaries archives. This adaptation does not reproduce all Boundaries research arms or independently replay every Adam update. Pause/stop are cooperative, so a finite qualification operation can finish before a pending control request is acknowledged. See `README.md`, `REFERENCE.md`, and the run receipts for exact scope and reproduction details.
