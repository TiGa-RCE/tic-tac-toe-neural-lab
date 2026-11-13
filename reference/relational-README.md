# Public recurrent relational policy

This package is a separate student architecture for the Boundaries learning lab.
It keeps the complete public 3×3 puzzle graph and learns to exchange messages
between objects, clue factors, and row factors. It imports neither the teacher
nor the independent verifier. Training labels, rewards, admission checks, and
final grading belong to the experiment driver outside this package.

The default network has **28,771 trainable parameters**, hidden size 32, and eight
shared recurrent rounds per proposed edit. Its native output contains nine
objects × three ordered columns. It can return logits after every round for
deep supervision. PyTorch is needed for training; portable inference needs only
NumPy and the standard library.

## Representation and inference boundary

`compile_puzzle` reads only the public format, ruleset, dimensions, clues, and
givens. Object identifiers index graph connections and row membership; they do
not become numeric identity features. Challenge hashes, seeds, family IDs,
solutions, certificates, and arbitrary private metadata cannot enter the model
batch. Puzzle certification is performed separately before admission.

Every clue is a typed factor. The encoder preserves all formal operand roles,
including the middle operand of Between and NotBetween and the anchor of XOR.
It distinguishes SameColumn with two operands from SameColumn with three.

| Factor type | Ordered formal operands retained |
|---|---:|
| SameColumn | 2 or 3 |
| NotSameColumn | 2 |
| SameColumnExcluding | 3 |
| XorSameColumn | 3 |
| NextTo | 2 |
| LeftOf | 2 |
| Between | 3 |
| NotBetween | 3 |
| Row permutation | 3 exchangeable members |

Rows use a shared mean-pooled factor rather than numeric row IDs or an arbitrary
ordering of their member identities. Given positions and editable estimates use
separate four-way channels: unknown, left, center, right. When one object is
queried, its previous editable estimate is masked; any immutable given remains
visible. No candidate legality mask, finite-clue-support computation, solver
call, solution search, or correction of a proposed action is performed.

Every recurrent round receives the original object and factor embeddings again.
Factor GRUs receive the current object states in their three formal operand
slots. Object GRUs receive the mean of incoming role-specific factor messages.
The shared readout produces three column logits. Hidden states reset on each
external call; the editable game state is the only history carried between
proposals.

Consistent renaming of objects within rows and consistent permutation of whole
rows permute the model outputs. **Column order remains fixed**: left, center,
right have different meanings. Clue order and graph padding do not affect the
computation except for floating-point rounding. The external fixed-index action
schedule is intentionally preserved; that schedule itself is not equivariant
under object renaming.

The historical identical-feature fixture in `examples/feature-collision.json`
has distinct relation incidence under this encoder. This removes that specific
input collision. It is not evidence that the model has learned either answer,
nor a guarantee that a finite message-passing network can solve every puzzle.

## Python API

```python
from relational.encode import compile_batch, build_batch, encode_batch
from relational.model import RelationalPolicy, to_torch

# For convenience: compile public puzzles and encode their current estimates.
batch = encode_batch(puzzles, estimates, query_objects=query_objects)

# For repeated training/evaluation: compile static graph structure once.
bank = compile_batch(puzzles)
batch = build_batch(bank, puzzle_indices, estimates,
                    query_objects=query_objects)

model = RelationalPolicy(hidden=32, rounds=8)
final_logits = model(to_torch(batch))                       # [B, 9, 3]
all_round_logits = model(to_torch(batch), return_all_steps=True)  # [T, B, 9, 3]
numpy_logits = model.predict(batch)                         # [B, 9, 3]
```

`estimates` is an integer array shaped `[B, 9]`, with values −1, 0, 1, or 2.
Omitting it starts from public givens and unknown cells. `query_objects` is an
optional integer array shaped `[B]`. With a graph bank, `puzzle_indices` chooses
and can repeat bank entries; estimates and queries correspond to that selected
batch. There is no step feature. Returned fields are exactly `objects`,
`factors`, `incidence`, and `factor_mask`.

For the controlled architecture comparison, each example contributes **one
queried-object label**, even when supervising all internal rounds:

```python
import torch
import torch.nn.functional as F

logits = model(to_torch(batch), return_all_steps=True)
query = torch.as_tensor(query_objects, dtype=torch.long)
rows = torch.arange(len(query))
scores = logits[:, rows, query, :]   # [T, B, 3]
loss = F.cross_entropy(scores.reshape(-1, 3), labels.repeat(model.rounds))
```

This is mean cross entropy across the same labels at eight internal rounds. It
does not introduce eight independent training examples or nine supervised
object labels per query snapshot. Labels must come from the separately managed
training corpus. A model can score all nine objects without those additional
objects being included in the loss.

## Honest sequential rollout

```python
from relational.rollout import run, run_batch

result = run(public_puzzle, model.predict, sweeps=3, retain_trace=True)
results = run_batch(public_puzzles, model.predict, sweeps=3)
# Grade result["submission"] with the independent verifier after rollout.
```

The primary schedule matches the previous student: objects 0 through 8, skipping
givens, repeated three times. With no givens this makes **27 proposed edits**;
eight internal rounds per edit mean 216 recurrent rounds. Each proposal is
applied even if it creates a collision. Later sweeps can overwrite earlier
errors. No teacher or verifier query occurs during rollout. The submission
serializer preserves incomplete or conflicting outcomes for external grading.

`run_batch` batches the same action slot across puzzles. It returns one result
per puzzle with `positions`, `submission`, `model_calls`, `teacher_queries`,
`verifier_queries_during_rollout`, and optional `trace`. Reported model calls are
per-puzzle decisions; batching reduces physical calls, not the action budget.
An optional `initial_positions` supports evaluation from a supplied editable
state. Public givens must agree before any action is taken.

## Portable weights and command line

```python
from relational.model import save_npz, load_npz
from relational.portable import load_policy

save_npz("relational-policy.npz", model)
torch_model = load_npz("relational-policy.npz")
portable_model = load_policy("relational-policy.npz")
results = run_batch(public_puzzles, portable_model.predict)
```

Ordinary PyTorch `state_dict` export is also supported. The portable NPZ format
stores float32 weights, the encoding version, hidden size, and round count. It
requires no pickle and rejects unexpected tensor keys, incompatible shapes,
nonfinite weights, and unknown formats. The NumPy implementation uses the same
GRU gate ordering and equations as PyTorch.

Run the standalone student from the lab directory:

```bash
python -m relational --model relational-policy.npz --puzzle public-puzzle.json \
  --output submission.json --trace rollout.json
```

Omit `--output` to emit only the official submission JSON on stdout. Use
`--puzzle -` to read public JSON from stdin. The optional trace records every
decision plus the inference configuration; it is not a verdict. The command
defaults to three sweeps and one CPU thread, setting BLAS thread limits before
NumPy is imported. `--threads` and `--sweeps` explicitly change those settings.
No PyTorch, training corpus, solver, or verifier is imported by this command.
Invalid input produces exit status 2 and an explanation on stderr.

## Verification and measured smoke cost

Run the focused tests using a Python environment with NumPy and PyTorch:

```bash
python -m unittest discover -s tests -p 'test_relational*.py' -v
```

Tests cover typed connectivity and roles, the prior input collision, private
metadata exclusion, given/guess separation, query masking, object renaming,
clue order, padding, immutable-input reinjection, real gradient updates, portable
weight validation, PyTorch/NumPy parity, honest sequential rollouts, and forbidden
inference imports. Command-line tests use an import guard and a deliberately
incorrect zero-weight policy to confirm that invalid proposals are emitted
without correction.

On the session CPU with PyTorch 2.15.0+cpu, NumPy 2.3.5, one PyTorch thread, hidden
32, and eight internal rounds, a smoke measurement using 512 public training
puzzles from the prior corpus (at most five clues) measured:

| Operation | Observed wall time |
|---|---:|
| Batch-128 Adam update with all eight queried-object rounds supervised | 30.51 ms median; 30.82 ms mean |
| 512 puzzles × 27 sequential edits, in chunks of 128 | 1.144 s total |
| One batch of 16, PyTorch inference | 1.90 ms |
| Same batch of 16, NumPy inference | 2.94 ms |

The update timing used three warmups and 20 measured updates; synthetic labels
were used only to measure computation. These timings are not a trained accuracy
result and can change with host load and larger graphs. The measured maximum
PyTorch/NumPy logit difference was 1.79×10⁻⁷, with identical argmax outputs in
that sample. Floating-point near-ties still need parity checks on any selected
trained artifact. This package does not claim whole-puzzle success rates; those
belong to the independently evaluated experiment outputs.
