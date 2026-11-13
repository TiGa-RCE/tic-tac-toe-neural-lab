# Boundaries reference and tic-tac-toe adaptation

The supplied Boundaries archive is the actual reference for this implementation. No prior Sudoku training artifact was located. The recipe below was observed directly in archive source and documentation. Historical accuracy and audit claims in the supplied documents have **not** been reproduced or independently verified here.

## Preserved reference sources

Archive: [Boundaries Frontier Lab 0.11.0, Part 1](</Volumes/PATRIOT/Downloads/Boundaries-Frontier-Lab-0.11.0-Part-1-of-2.zip>). Original files remain unchanged. The small copies below preserve complete members except for the explicitly identified configuration excerpt. SHA-256 identifies the bytes read; it does not validate historical outcomes.

| Archive member (under `Boundaries-Frontier-Lab-0.11.0/`) | Local reference | Original member SHA-256 | Size | Scope |
| --- | --- | --- | ---: | --- |
| `docs/TRAINING-V11.md` | [TRAINING-V11.md](reference/TRAINING-V11.md) | `db4fda6c75a76be1fdcc278dd4c00a594ce676cd32552f90546e0791cd8c68f0` | 9,178 bytes | complete member, unchanged |
| `relational/README.md` | [relational-README.md](reference/relational-README.md) | `56061b65b2dede2769ba10b7fbe3ff7f1dd246ee3b5ba2a5c04a300ba80dc1f4` | 9,748 bytes | complete member, unchanged |
| `consolidation/prepare.py` | [consolidation-prepare-config.py.txt](reference/consolidation-prepare-config.py.txt) | `3a0093c4323cdb0338e89e9cc52acc26b0a5e34bc6ccd8e98acbbe730fc6ab6f` | 5,692 bytes | source lines 21–43; excerpt SHA-256 349e6ea26b97f66d44d162cb4d21c9de458060da7135e313bdbaccb972ea8159 |

## Observed recipe

The earlier consolidation configuration uses optimizer seeds 11, 23, and 37; initial expert examples and random corruptions; DAgger collection; diagnostic error groups; optional short targeted review or MIR; and validation checkpoint selection including the starting model. It keeps Adam state through each continued trajectory. Optional policy-gradient updates appear in the warm continuation arms; the recurrent architecture comparison uses supervised CE without RL.

The V11 continuation recipe has three phases of 512 Adam updates per branch. Before each phase it collects the current student's native TRAIN trajectories, then attaches certified TRAIN labels. Every update balances eight native and eight retention complete-state groups. Its objective combines mean CE with `0.1 * T² * KL(parent || candidate)` at temperature 2, restricted to TRAIN queries where the frozen parent's greedy answer is correct. It clips the global gradient norm at one, saves the literal phase schedule before execution, and preserves optimizer states, weights, losses, collection traces, and phase checkpoints. Validation measures both repairs and broken incumbent successes; final test follows frozen selection.

The Boundaries relational student uses hidden size 32 and eight shared recurrent rounds, totaling 28,771 parameters. Its runtime imports neither teacher nor verifier; portable inference uses NumPy. The proposal model does not repair its predictions using a solver. Those architecture choices are domain-specific; the transferable recipe is solver supervision, honest native-state collection, retention-aware review, explicit selection, frozen qualification, and portable inference checks.

## Chosen tic-tac-toe adaptation

This is a new, deliberately smaller experiment grounded in that recipe. It does not claim to reproduce the historical Boundaries system.

| Stage | Planned tic-tac-toe behavior | Relation to the reference |
| --- | --- | --- |
| Solver and admission | Enumerate reachable 3×3 states; deterministic minimax labels all equally optimal legal moves. Partition by D4 board orbit into TRAIN, META, validation, and TEST. | Exact teacher plus explicit data boundary; D4 grouping prevents rotated/reflected split leakage. |
| Founder distillation | A `27 → 64 → 64 → 9` tanh policy MLP; 1,024 supervised optimal-set CE updates. | Small shared student trained by deterministic teacher; architecture selected for this game. |
| Native collection and review | Seeds 11/23/37 each receive three phases of 512 updates, honest TRAIN-only native rollouts, and half native/half retention sampling. | Native collection, balanced retention, phase checkpoints, and multiple declared seeds. |
| Retention objective | Adam at `3e-4`, global gradient clipping at one; CE plus `0.1 * T² * KL(parent || candidate)`, `T=2`, on parent-correct TRAIN states. | Retains V11 objective shape while choosing a new learning rate for the smaller task. |
| Reproducibility | Preserve full literal schedules, native collection records, phase optimizer states, losses, and checkpoints. | Inspectable phase evidence rather than progress inferred from elapsed time. |
| Selection | Include founder; rank candidates by validation repairs with breaks explicitly counted; use META for finalist tie checking before sealing. | Development-only selection and a retained starting candidate. |
| Qualification and export | Seal the selected policy; run TEST and exhaustive reachable-state audit without retuning; compare training and portable inference. | Frozen evaluation and explicit numerical/behavioral checks. |

Gameplay uses the selected frozen neural network. The deterministic solver supplies training labels and external evaluation, not move selection during gameplay. Ordinary game legality checks remain separate from strategic solver assistance. Exhaustive reachable-state audit is finite-domain coverage, not evidence of generalization to another board size or game. If all states later enter training, that must be disclosed separately from the original held-out result.

The architecture, update counts, split proportions, and selection details above describe the chosen implementation contract. Actual run receipts and implementation documentation determine what executed and its measured results; this reference note alone is not evidence of completion or accuracy.
