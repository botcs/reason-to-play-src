# Reproduction limits

This record distinguishes supported analyses from unresolved inputs needed
for exact paper reproduction. The dataset manifest identifies available files,
checksums and source attribution. A matching model name or filename alone does
not establish the checkpoint and code revision that produced a result.

The paper distinguishes eight behavioral LRMs from seven encoding LRMs.
The larger Qwen behavioral condition is not an additional headline encoding
model. Earlier first-person features, later multi-turn features, prompt variants,
temporal controls and random-initialization runs must remain separate conditions.
The reproduction guide's fresh action-only/minimal example demonstrates an
entrypoint; it does not assert that this was the headline paper condition.

## Headline encoding conditions

The headline analysis uses the following model/variant labels, with stream
`main` and performance band `main`. Layer counts describe the configured
decoder layers; they do not assert complete artifact coverage. These labels
do not establish exact historical weight revisions.

| Model repository | Analysis model ID | Variant | Layers |
| --- | --- | --- | ---: |
| `Qwen/Qwen3.5-9B` | `qwen35_9b` | `all` | 32 |
| `Qwen/Qwen3.5-27B` | `qwen35_27b` | `all` | 64 |
| `Qwen/Qwen3.5-35B-A3B` | `qwen35_35b_a3b` | `all` | 40 |
| `Qwen/Qwen3.5-122B-A10B` | `qwen35_122b_a10b` | `all` | 48 |
| `deepseek-ai/DeepSeek-V3.2` | `dsv32` | `compressed` | 61 |
| `deepseek-ai/DeepSeek-V4-Flash` | `dsv4_flash` | `compressed` | 43 |
| `deepseek-ai/DeepSeek-V4-Pro` | `dsv4_pro` | `compressed` | 61 |

The headline encoding cohort is the 21 vgfmri4 participants. Keep model,
variant, stream, performance band and nuisance-fit condition explicit when
selecting tables or regenerating results. The separate dataset manifest records
which individual source objects are present.

## Baseline checkpoints and layer mappings

The [baseline guide](guides/baselines.md) documents the bundled DDQN and
EfficientZero implementations, source revisions and training setup. File hashes
and adaptation notices identify the included implementations. New baseline
runs use these sources without requiring a private RC_RL checkout. EfficientZero
feature extraction uses the included `agents/efficientzero/inference/` and
`environment/` code. Its optional `training/` submodule pins Austin Andrews's
trainer to `29157d4892afd9467b1bd0994de1355086145490`. That pin identifies the
training source, not the revision that produced every released checkpoint.
Initializing the submodule is unnecessary for analysis or feature extraction;
a full retraining run has not been validated.

The EfficientZero extractor requests **15 named hooks**. The archived encoding
job configuration selects **11**: four representation hooks and seven initial
value/policy hooks. The result summary uses eleven numeric layer labels, but
their exact mapping to the named hooks remains unverified. The released
alignment command samples the named hooks at the existing BOLD samples;
it does not establish which arrays produced the archived scores.
Likewise, the saved DDQN channels cannot automatically be identified
with every historical result-layer label. New baseline ROI rows require an
explicit layer map; the code does not relabel candidates as final paper inputs.

Available checkpoint/configuration candidates come from different generations.
Some actual EfficientZero training configurations use RGB observations and a
13-level curriculum, differing from the manuscript's baseline description.
Some reported checkpoint paths have no corresponding weights among the
available artifacts. Successful extraction
with an available checkpoint establishes code compatibility, not final-paper
checkpoint identity or an exact reproduction of the reported baseline.

## Coverage and analysis conventions

Two headline result cells are missing from both the available NPZ results
and the master CSV: Qwen3.5-27B/sub-20/layer23 and
DeepSeek-V3.2/sub-31/layer61. Some raw feature tensors are also absent despite
available aligned features and results. These gaps remain explicit;
absence is never encoded as zero performance. A temporal-shuffle caption and
the stored model/layer labels also need reconciliation with the original
figure-producing inputs. Available result values retain their recorded scores.

Human play outcomes are nullable and death events must be checked before
classifying incompletes. Practice runs, cohort-specific upper levels and
participant-versus-model timeout differences matter for the documented
comparison rules. Humans advanced on a fixed scanner schedule; model agents
used blocked advancement. Keypress counts, model decisions and engine frames
are different clocks. EMPA1 is the reported comparison; EMPA2 is excluded by
default. See the [reproduction guide](reproducibility.md) and
[analysis methods](analysis-methods.md).

Neural outputs distinguish feature variant/stream, performance band and fit
condition. Main-only and nuisance-adjusted fits must not be silently combined.
Best-layer selection and stochastic fitting follow the documented analysis
conventions. Executable code alone does not independently validate the paper’s
conclusions.

## Validation scope

The [validation summary](release/validation.json) records test commands and
execution limits. The human-file checks cover all 6,994 task attempts and
1,661,744 recorded states. Measured trajectories and play and frame IDs agree
across all three prompt conditions; the catalogue counts each observation once.
Representative checks cover recorded human replay,
three-stream extraction through a tiny synthetic CPU model, baseline runtime
checks, and synthetic NIfTI-to-ROI integration. The tiny model validates the
extraction interface and timestamps; its features are not the activations used in the paper. Figure commands also run against the corresponding result tables;
the code repository does not embed participant-derived tables.

All 411 selected LRM feature archives were realigned from their corresponding
source generations and the human clocks: all 20,081 arrays agree exactly with
the saved scanner-sampled features. The 21 HRR archives also agree exactly.
The manifest pins the input generations. Each LRM association records missing
play intervals separately; these total 5,727 of 603,513 sample instances across
the archives, counting a participant's samples again for each model or condition.
The original zero-filled intervals remain explicit. Exact alignment does not
mean every retained sample has an observed model feature.

Further checks reproduce 6,426 archived ROI rows with the released atlas and
32-participant common mask, and repeat a fresh seeded fit over all 1,470 samples
of one participant with a bounded 16-voxel target set. The two fresh fits are
identical; they do not claim historical-score or whole-brain reproduction.

EfficientZero alignment covers 11 named hooks for all 32 participants and
46,581 retained samples. Independent frame-clock reductions agree exactly with
all 73,062 play/layer slices in the final archives. The raw inventory lacks nine
of the 6,994 task plays; none belongs to the 6,642-play BOLD selection, so these
aligned archives have complete source coverage. Two seeded fits using the
participant-13 policy features, all 1,470 samples and 16 BOLD voxels also agree
exactly, with network and source-service imports disabled. These checks support
new encoding fits; the historical baseline layer/checkpoint attribution remains
unresolved as described above.

The explicit archived-table selection also generates the regional encoding
figure from 1,169,226 main-band rows across ten model labels and 21 participants,
using the dataset tables without source archives or network connections. The
selection pins the
source table's checksum and acknowledges the two known absent cells. It keeps
unrecorded fit conditions and unlabeled baseline streams unknown; figure
generation alone does not identify those upstream settings.

Validation does not cover full participant fMRIPrep, whole-cohort fits,
large-model GPU/distributed extraction, full EfficientZero training or a complete
dataset upload. See the [manifest workflow](release/manifest-guide.md) for catalogue schemas and
file verification, and the [dataset card](release/huggingface-dataset-card.md) for
publication status.
