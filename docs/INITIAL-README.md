# Three in a Row — Neural Training Lab

Open **http://127.0.0.1:8765** while the local server is running. Play as X or O against a frozen small neural network, inspect its policy probabilities, and optionally compare its choices with the exact teacher. The teacher overlay never chooses a gameplay action. Starting a new game loads the latest checkpoint; an ongoing game's tensors stay fixed.

## Run

```bash
cd /Volumes/CODEX/Codex/2026-11-13/can-you-make-me-a-tic/outputs/tictactoe_lab
python3 server.py --port 8765
```

Or double-click `Launch.command`. Stop the server with Ctrl-C. It binds only to loopback. Python 3.10+, NumPy, and Node 18+ are required for training and export verification. No JavaScript packages, build step, API keys, network requests, or hosted services are required. The current Mac already supplies these dependencies. `requirements.txt` records the only Python package.

The dashboard's **Start training** creates a new run, checks storage for a conservative 100 MB allocation, and launches training through this Mac's supported `memory-run` with a 512 MiB total peak budget. Those two local safety tools must exist; the server refuses training if they are unavailable. The worker runs on CPU with one numerical thread. Pause/resume preserve live optimizer state; stop is cooperative at update/phase boundaries. Server restart preserves evidence and checkpoints but does not resume an orphaned worker. Completed runs remain under `runs/`; no automatic deletion occurs.

## Chosen recipe

The supplied Boundaries 0.11.0 source is the verified reference; see [REFERENCE.md](REFERENCE.md) and the small preserved extracts. A separate Sudoku reference was not found. This is a scaled adaptation, not a reproduction of the original recurrent multi-expert system or its historical results.

1. **Teacher qualification.** Enumerate all 5,478 legal reachable states, stop after wins, and obtain exact minimax labels for the 4,520 decision states. Accept every equally optimal move. Split the 627 decision-state symmetry orbits into 70% TRAIN and 10% each META, validation, and TEST; rotations/reflections cannot cross partitions. Independently test the teacher against an absolute-X max/min oracle.
2. **Supervised foundation.** A 27 → 64 → 64 → 9 tanh MLP has 6,537 parameters. Inputs are cell-major `[own, opponent, empty]` one-hot features from the next player's perspective. Outputs are policy logits. Optimize negative log probability mass on the entire optimal-action set. Adam uses 1,024 updates, batch 128, learning rate 0.003, global norm clipping at 1.
3. **Collect → review → retain, three phases.** Seeds 11, 23, and 37 share the same founder; they are sampling replications, not independent pretrained lineages. Each branch starts fresh Adam, then preserves it across three phases of 512 updates at 0.0003. Native neural rollouts start in TRAIN and stop at the partition boundary. Exact labels never override native moves. Save the full literal schedule before execution: 64 native samples plus 64 TRAIN retention samples per update, with 25% targeted current TRAIN errors for the first 32 updates when any remain. Add `0.1 × T² KL(parent || candidate)`, T=2, only where the frozen founder is correct. Keep native multiplicities, traces, Adam state, weights, loss logs, and phase repair/break counts.
4. **Selection.** Include the unchanged founder. Rank by validation repairs, fewer validation breaks, then validation successes. META breaks exact finalist ties; earliest candidate breaks remaining ties. Seal the selected checkpoint and protocol hashes before TEST labels are accessed. Raw-final and selected models remain separately identifiable.
5. **Qualification and export.** Evaluate the frozen selection once on TEST within that run. Then report full-domain coverage separately. An independent standard-library verifier imports neither teacher nor trainer, implements another oracle and forward pass, grades all 4,520 positions, and explores every possible opponent continuation from empty as X and O. Compare the browser module against all independent logits and actions at a fixed 1e-9 absolute tolerance. No post-test optimization or checkpoint substitution occurs.

The 9-action game uses occupied-cell masking as its ordinary action interface. There is no tactical win/block rule, minimax fallback, solver table, or learned rule critic in browser inference. Policy probabilities are **not win probabilities**. A small student can still make mistakes; see the measured qualification panel and `counterexamples.json`.

## Evidence and interpretation

Each run records `protocol-before-training.json`, `corpus.json`, literal schedules, native traces, founder and all nine branch checkpoints (JSON plus parameters/Adam in NPZ), `selection-before-test.json`, immutable `test-evaluation.json`, `test-consumption.json`, `evaluation.json`, independent fixtures, portable parity, metrics, events, and exact source snapshots. JSON exports can run without Python or NumPy using `web/network.js`.

The original `initial-seed42` smoke run is retained with its original source bytes. Review found that its native collector and parent inference could query excluded inputs, although every optimizer sample was TRAIN and no TEST label entered optimization. The final collector stops at the partition boundary, and the parent cache queries TRAIN only. The final run uses unchanged architecture, hyperparameters, selection criteria, and seed. Its repeated finite partition is a diagnostic, **not a fresh independent scientific TEST campaign**. The original seal's `test_consumed:false` describes its time of creation; completed current runs have a separate hash-bound consumption receipt rather than rewriting the seal.

Exhaustive decision accuracy includes training positions and must not be described as held-out generalization. `no_loss_from_empty` is a separate stronger gameplay check for this exact frozen policy on the ordinary 3×3 game, and may fail even when decision accuracy is high. Neither a successful training exit nor portable numerical agreement proves a perfect policy.

The adaptation omits Boundaries-specific recurrence, relational factors, residual adapters, MIR, policy-gradient arms, learned critic routing, multi-actor runtime recovery, and independent replay of all optimizer updates. The local qualification includes a finite-difference gradient check, solver tests, artifact/partition audits, independent inference, and full exported-action parity.

## Checks

```bash
python3 -m unittest discover -s tests -v
```

These ordinary checks perform no neural inference. The completed trainer already runs independent model qualification and browser parity inside its guarded process. To deliberately re-evaluate an export separately, use `memory-run` as required by the local policy:

```bash
/Users/tiga/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 verify_export.py --model runs/RUN/model.json --output runs/RUN/recheck.json
```

Results are specific to the saved source, split, seed, checkpoint, numerical backend, and fixed tie-breaking policy (lowest cell index on equal logits).
