# Three in a Row — Neural Training Lab

**[Play the public ChatGPT Site](https://three-in-a-row-neural-lab.tiga-rce.chatgpt.site)** · **[Source, weights and evidence](https://github.com/TiGa-RCE/tic-tac-toe-neural-lab)**

A **6,537-parameter neural network** plays ordinary 3×3 tic-tac-toe. A deterministic minimax teacher supplied training labels. Gameplay is a forward pass through frozen neural weights, occupied-cell masking, and a stable argmax. **There is no solver, tactical rule, lookup-table move selector, or fallback during play.**

The published student is optimal on **all 4,520 reachable decision states**. An independent oracle and inference implementation also checked every possible opponent continuation from empty: worst-case **draw as X and draw as O**. The frozen policy is therefore unbeatable on the ordinary game. Every decision state was deliberately supervised during continuation; this is exhaustive finite-domain qualification, not held-out generalization.

## Public dashboard

Play either side, inspect policy probabilities, explicitly compare with the teacher, download weights, and inspect actual phase curves, checkpoint repairs/breaks, and qualification results.

The initial dashboard shows **saved published evidence**. **Train in this browser** starts real CPU training in an isolated Web Worker on the visitor's device: three phases of 512 Adam updates, native collection, targeted review, balanced retention, and parent-correct temperature-2 KL. Pause/resume/stop are cooperative; literal schedules and run evidence can be downloaded. Visitor runs do not modify the hosted baseline. The game retains the independently certified neural checkpoint throughout experimental training.

Teacher labels load only after explicit training or enabling the optional comparison overlay. Default gameplay loads only the neural weights and run summary, without fetching the teaching corpus. Policy probabilities are not win probabilities.

## Run locally

The complete published experience needs no Python package or build step:

```bash
git clone https://github.com/TiGa-RCE/tic-tac-toe-neural-lab.git
cd tic-tac-toe-neural-lab
python3 -m http.server 8000 --directory site/dist
```

Open `http://localhost:8000`. Browser training is available too.

The Python dashboard controls local numerical training and writes durable receipts:

```bash
python3 -m pip install -r requirements.txt
python3 server.py --port 8765
```

Open `http://localhost:8765`, or use `Launch.command` on macOS. Python 3.10+, NumPy and Node 18+ support the numerical trainer/export qualification. The public version needs only a modern browser.

The local controller requires supported storage admission and memory supervision tools, defaulting to `~/.codex/bin/storage_admit.py` and `~/.local/bin/memory-run`. `TTT_STORAGE_ADMISSION` and `TTT_MEMORY_GUARD` can select supported equivalents. Training fails closed if these tools are unavailable. It admits 100 MB of growth and launches a single-thread CPU worker with a conservative 512 MiB peak budget. No GPU, API key or external model download is used. Stop the server with Ctrl-C; old runs are never automatically deleted.

## Chosen recipe

The supplied Boundaries 0.11.0 source grounds the method; [REFERENCE.md](REFERENCE.md) and `reference/` preserve small source extracts. The Sudoku reference was not available. This scales the methodology to a feedforward game policy rather than reproducing Boundaries' recurrent multi-expert system.

1. Enumerate 5,478 reachable boards and 4,520 decision positions. Validate the teacher against an independent absolute-X max/min oracle. Supervise every equally optimal move using negative log total policy probability on the optimal-action set.
2. The initial study uses D4 orbit-disjoint TRAIN/META/validation/TEST partitions and a 27 → 64 → 64 → 9 tanh MLP with cell-major `[own, opponent, empty]` features. Foundation: 1,024 Adam updates, batch 128, learning rate 0.003, norm clipping at 1. Its imperfect selected model and all prior evidence remain intact.
3. The continuation starts from that selected student and explicitly promotes all 4,520 states to TRAIN. Seeds 11, 23 and 37 share the parent, with fresh Adam preserved across phases. Each phase executes a previously saved 512-update schedule at learning rate 0.001: 64 native-multiplicity samples plus 64 uniform retention samples. The first 32 updates use 25% targeted current errors/adversarial critical states where needed. Teacher labels never override native actions.
4. Retention uses `0.1 × T² KL(parent || candidate)`, T=2, only where the frozen parent is correct. Save traces, schedules, losses, tensors, Adam state, source identities and repairs/breaks. Rank no-loss coverage as both sides, then optimal decisions, fewer breaks, and earlier checkpoints. Freeze selection before independent qualification.
5. Four complete three-seed phases — **6,144 continuation updates** — reached mastery in all three branches. The registered ranking selected `s11-p4`. Independent qualification grades every decision position and every adversarial continuation as X and O; exported browser actions/logits must agree at fixed absolute tolerance 1e-9.

The model still has 6,537 parameters. No ensemble, learned critic, per-position correction table or solver fallback was added.

## Evidence and reproduction

Final run: `runs/20261113T233235Z-7e5be6/`. It contains protocol/corpus, literal schedules, native traces, source snapshots, parent and all twelve branch checkpoints, selection seal, qualification receipt, `model.json`, independent fixtures, and portable parity. `browser-worker-verification.json` exercises the identical worker through a Node adapter: all 1,536 updates, pause/resume, and certified-incumbent preservation.

See [VERIFICATION.md](VERIFICATION.md). Earlier machine paths in immutable historical receipts are provenance, not runtime requirements. Original smoke and split-study evidence remain preserved. The external Boundaries ZIP archives and unrelated user files are excluded.

```bash
python3 -m unittest discover -s tests -v
python3 build_site.py --run-dir runs/20261113T233235Z-7e5be6 --output site/dist
```

The ordinary tests do not load learned tensors. Deliberate requalification should use the supported local guard:

```bash
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 verify_export.py --model runs/20261113T233235Z-7e5be6/model.json --output recheck.json
```

The static Site needs no server-side secrets, account connectors, training service, or connection to the original Mac.
