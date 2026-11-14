# Three in a Row — Neural Training Lab

**[Play the public ChatGPT Site](https://three-in-a-row-neural-lab.tiga-rce.chatgpt.site)** · **[Source, weights and evidence](https://github.com/TiGa-RCE/tic-tac-toe-neural-lab)**

The default game uses the **49-coefficient sparse fully optimal network**: it chooses an optimal move on **all 4,520 reachable decision boards** and has a worst-case draw as both X and O against every possible opponent continuation. Its float16 coefficient values pack into **98 bytes**; JSON metadata, fixed geometry, and runtime code are additional. Inference uses ordinary JavaScript double arithmetic.

The same four-neuron network scores all eight rotations/reflections; mapped logits are averaged before occupied-cell masking and a declared numerical argmax rule. Eight readouts plus a fixed ninth zero and shared geometric biases eliminate redundant coefficients. The deployed format contains 49 learned numeric values, 24 structural zeros and one fixed unit; structural metadata is additional storage. **No solver, tactical feature, per-position move table, or fallback chooses a move during play.**

The **Opponent network** menu lets you choose the default 49-coefficient sparse model, the preserved 53- and 74-coefficient fully optimal models, the earlier 38-coefficient shared model, the 123-parameter dense model, or the original 6,537-parameter perfect model. Changing the opponent starts a new game with that frozen network. The 38-coefficient model remains unbeatable from empty and is optimal on 3,888/4,520 arbitrary decision boards (86.02%).

The 49-coefficient model is the **smallest fully optimal result verified in the retained bounded searches**, not a proof of a global minimum. It reduces both learned coefficients and numeric payload by **7.55%** from the 53-coefficient model. See [SPARSE-FOLLOWUP.md](SPARSE-FOLLOWUP.md) for masked fitting, bounded support search and quantization-aware margin refinement. The earlier [53-coefficient study](SPARSE.md) and [74-coefficient study](PERFECT.md) remain preserved. The 38-coefficient model is the **smallest verified no-loss result found in the earlier 71 bounded training trials plus algebraic folding**. See [COMPRESSION.md](COMPRESSION.md) for that search and its preserved strict-parity failure.

## Public dashboard

Play either side, inspect policy probabilities, explicitly compare with the teacher, download weights, and inspect actual phase curves, checkpoint repairs/breaks, and qualification results.

The dashboard shows **saved published evidence** for the default network; the complete search receipts remain in this repository. The **Training network** menu offers the default **49-coefficient sparse fully optimal network**, the preserved **53- and 74-coefficient fully optimal networks**, the 38-coefficient shared network, the 123-parameter one-pass dense model, and the original 6,537-parameter perfect model. Each choice displays its all-board optimality coverage. The training selection is independent of **Opponent network** and changes the visitor experiment only. **Train in this browser** starts real CPU training in an isolated Web Worker on the visitor's device: three phases of 512 Adam updates, native collection, targeted review, balanced retention, and parent-correct temperature-2 KL. Pause/resume/stop are cooperative; literal schedules and run evidence can be downloaded. Visitor runs do not modify the hosted baseline. The game retains the selected, independently certified neural checkpoint throughout experimental training.

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

That original continuation retained all 6,537 parameters. The later [compression study](COMPRESSION.md) and [fully optimal compact search](PERFECT.md) retain the same teacher/native collection/review/retention recipe while changing network size and the acceptance gate. The D4 variants reuse one network over eight fixed geometric transforms; they do not train separate ensemble members. The [first sparse study](SPARSE.md) adds coefficient pruning and linear readout fitting; the [further sparse study](SPARSE-FOLLOWUP.md) adds masked full-domain optimization, bounded support search and margin fitting with quantized hidden weights. These optimizers run only during training. No learned critic, per-position correction table or solver fallback chooses gameplay actions.

## Evidence and reproduction

Earlier no-loss compact export: `runs/compression-20261113-selected/`. Default sparse fully optimal export: `runs/perfect-sparse-49-refined/`, selected by `runs/sparse-perfect-selection.json`; the previous `runs/perfect-sparse-53/` remains selectable. Preserved dense fully optimal export: `runs/perfect-optimal-74/`, selected by `runs/perfect-optimal-selection.json`; its 92-coefficient predecessor is preserved in `runs/perfect-optimal-92/`. Original perfect run: `runs/20261113T233235Z-7e5be6/`. It contains protocol/corpus, literal schedules, native traces, source snapshots, parent and all twelve branch checkpoints, selection seal, qualification receipt, `model.json`, independent fixtures, and portable parity. `browser-worker-verification.json` exercises the identical worker through a Node adapter: all 1,536 updates, pause/resume, and certified-incumbent preservation.

The 49-coefficient sparse choice passed a complete 1,536-update browser-worker run with pause/resume, preserved structural zeros and fixed units, and retained the certified 4,520/4,520 training parent. The earlier 53- and 74-coefficient worker results remain preserved too. See [SPARSE-FOLLOWUP.md](SPARSE-FOLLOWUP.md), [SPARSE.md](SPARSE.md), [PERFECT.md](PERFECT.md) and [VERIFICATION.md](VERIFICATION.md). Earlier machine paths in immutable historical receipts are provenance, not runtime requirements. Original smoke and split-study evidence remain preserved. The external Boundaries ZIP archives and unrelated user files are excluded.

```bash
python3 -m unittest discover -s tests -v
python3 build_site.py --run-dir runs/perfect-sparse-49-refined --output site/dist
```

The ordinary tests do not load learned tensors. Deliberate requalification should use the supported local guard:

```bash
~/.local/bin/memory-run --kind model --peak-bytes 536870912 -- python3 verify_export.py --model runs/perfect-sparse-49-refined/model.json --output recheck.json
```

The static Site needs no server-side secrets, account connectors, training service, or connection to the original Mac.
