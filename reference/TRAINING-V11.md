# V11 deeper learning and endpoint recovery

## Registered purpose

These two experiments follow the V10 evidence that trained sixteen-round
execution was still improving, and that many remaining failures were wrong
trajectories shared by the deployed roster. They produce candidate experts for
a separately selected controller. Their development selection does not claim
that either expert is a superior standalone policy.

The original V10 archive is unchanged. V11 uses fresh admitted TRAIN, META,
validation and test families, excluding the complete historical family union.
This training module can open only TRAIN, META and validation public inputs. It
has no test-evaluation command. Only `private/train-labels.jsonl` supplies
optimization labels. `results/frontier-v11/selection-before-test.json` closes
optimization when the coordinator freezes final selection.

## Two arms

| Arm | Initial tensors | Native training/inference purpose | Recurrent rounds |
| --- | --- | --- | ---: |
| D | V10 `M-s23-u1536`, SHA-256 `16815d0920e14a293b0b8583d27d3ec4a2a42d56326b9f69a456e8c421759f34` | Continue the previously trained deeper expert; start unknown or collapsed to column zero | 16 |
| R | V9 parent, SHA-256 `ab5882590b6081bf6f539d3a9bf2b573ec7981c6ebfadb0fffb3992bf97145a0` | Learn a further 27 greedy edits beginning from the frozen V10 system's completed endpoint | 8 |

Both preserve the original `RelationalPolicy` architecture and portable NPZ
envelope. No new learned parameter blocks, legality projection, exact solver,
verifier-driven action, or altered action masking are introduced. R receives a
publicly computed incumbent endpoint as its editable state; labels do not enter
that state at deployment.

Each arm uses optimizer seeds 11, 23 and 37. Every branch runs three phases of
512 updates, or 1,536 updates: six branches and **9,216 optimizer updates** in
total. All three seeds within an arm start from the same weights. They are
sampling/optimizer replications, not independently pretrained lineages.

Adam starts fresh for each branch with learning rate `1e-4`, its default betas
and epsilon, zero weight decay, and global gradient-norm clipping at 1. The old
Adam state is not resumed. Each update has eight native groups and eight
retention groups. A group is a complete pre-action state plus all nine correctly
masked query views, so every update contains 144 native query presentations.

Cross-entropy is averaged over all queried objects and all actual recurrent
rounds. The only auxiliary objective is `0.1 * T² KL(parent || candidate)` at
temperature 2, restricted to TRAIN queries where the frozen V9 parent's greedy
answer matches the certified TRAIN label. The parent cache uses eight rounds on
exactly the same complete prestate and queried view. This is a retention
control, not a guarantee of unchanged behavior.

The complete update budget contains 1,327,104 training query presentations and
15,925,248 recurrent forward query steps, with matching backward participation.
Native collection, parent-cache inference and development evaluation are extra
work and are recorded separately. D and R do not form a matched-pool factorial
comparison: their initial weights, depth, state distributions and intended
deployment roles deliberately differ.

## Data construction and literal replay

The common retention pool has six teaching contexts per TRAIN family: unknown,
all column zero, the certified solution, one seeded solution prefix from each
of the two ordinary starts, and one seeded random estimate vector. Prefix
lengths range from one through eight. These are supervised training contexts;
they do not grant the runtime access to a solution. Complete state duplicates
are merged while retaining their observation multiplicity.

D's initial native pool records all 27 pre-action states for both starts on all
TRAIN families using the original trained sixteen-round expert. Before each
phase, D adds a fresh native rollout on 256 seeded TRAIN families, half unknown
and half collapsed. Its current learned weights generate these actions.

R first executes the unchanged V10 sixteen-member system on TRAIN. The verifier
grades completed outputs only after inference. Its failed endpoints become
training starts. Each failed start also gets one seeded restart with a single
estimate masked unknown. R records native 27-action trajectories from both
versions, initially using the V9 parent and before each subsequent phase using
the current R model. No teacher overrides a recorded native action. Correct
incumbent cases remain represented through the common retention contexts,
including complete certified solutions.

The pool keys are `(TRAIN puzzle index, complete unmasked prestate)`. Query
masking occurs exactly once during batch construction. Repeated states preserve
their frequencies and the first occurrence order; conflicting labels are an
error. This avoids the earlier project's double-masking hazard.

Before each phase's first update, the sampler writes its full literal schedule:
512 rows of eight native indices, eight retention indices, and the declared
recurrent depth. Sampling is with replacement, weighted by the recorded
multiplicities. Subsequent execution reads that saved schedule. Parent caches
append only genuinely new state groups; old prefixes remain byte-identical.

## Development selection

Each branch preserves its original expert and the checkpoint after each phase.
All four candidates are evaluated on validation in both initial conditions.
R starts from the *fixed V10 system endpoint* in each condition; D starts from
the original condition. Grading happens after each native trajectory completes.

The predeclared branch selection key is:

1. Most validation failures of the incumbent repaired across both starts.
2. Most total validation successes.
3. Fewest invalid final boards.
4. Earliest checkpoint, including the untrained initial expert at update zero.

This selects a complementary expert. It can select a model with worse standalone
accuracy, so every candidate records new repairs, lost incumbent successes,
unchanged successes and unchanged failures. The selected expert then receives
one META evaluation for the coordinator's controller-selection work. The raw
last checkpoint remains recorded separately from the selected checkpoint.

The R evaluation receipt counts the 27 additional recovery edits, excluding the
already computed incumbent. Any deployed controller must include both pieces
of work. A successful process exit means inference finished; the verifier
outcome determines whether the proposal solved the puzzle.

## Qualification and audit

`qualification-before-training.json` captures six focused tests and the exact
qualified source hashes. They exercise grouping/masking, multiplicity and label
consistency, TEST-input rejection, native rollout equivalence with the original
export runtime at eight and sixteen rounds, two independent Adam updates, and
the declared checkpoint ranking. The protocol cannot register against a failed
qualification or changed source bytes.

`protocol-before-outcomes.json` binds this recipe, original weight identities,
corpus manifest/admission, TRAIN inputs, starter policy specifications and
source hashes before learned V11 outcomes. Every phase retains its collection
checkpoint, native prestates/actions, grouped pool, parent logits, literal
sampling schedule, weights, Adam state, loss log and evaluation evidence.

The first two actual updates of every phase additionally retain a before-state,
complete input capsule and after-two state. The audit reconstructs each batch
from its literal schedule/cache, verifies its targets against the TRAIN table,
and replays the objective using a separate log-softmax/gather calculation. The
predeclared maximum absolute parameter/optimizer-state tolerance is `2e-7`.
This covers 36 actual updates once all eighteen phases complete. It does not
independently replay every update or reimplement the shared relational encoder.

## Commands

From the V11 lab root, use the qualified Python environment and one numerical
thread:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m unittest tests.test_v11_training -v
python -m frontier.training register
python -m frontier.training prepare
python -m frontier.training train D 11
python -m frontier.training train D 23
python -m frontier.training train D 37
python -m frontier.training train R 11
python -m frontier.training train R 23
python -m frontier.training train R 37
python -m frontier.training summarize
python -m frontier.training audit --output results/training-v11/training-audit.json
```

The original preparation requires the already recorded qualification receipt
and fully admitted V11 corpus. Registering or retraining after the final seal
is rejected. A separate reproduction should preserve the delivered evidence
and generate its own qualification/protocol receipts before execution. Read
`results/training-v11/summary.json` and the six branch receipts for outcomes;
this document specifies the experiment and does not predict its success.
