# Scientific provenance and reproduction limits

This summary accompanies the consolidated code. Detailed storage audits,
participant catalogues, source-generation identifiers and historical result
tables are prepared separately and are not distributed in this Git release.
A recovered implementation or matching filename does not establish which
checkpoint and code revision produced a published result.

The paper distinguishes eight behavioral LRMs from seven encoding LRMs.
The larger Qwen behavioral condition is not an additional headline encoding
model. Earlier first-person features, later multi-turn features, prompt variants,
temporal controls and random-initialization runs must remain separate conditions.
The reproduction guide's fresh action-only/minimal example demonstrates an
entrypoint; it does not assert that this was the headline paper condition.

## Retained headline encoding conditions

The retained analysis uses the following model/variant labels, with stream
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

## Baseline lineage and unresolved mappings

The [baseline guide](../../baselines/README.md) records the bundled RC_RL roles,
EfficientZero source revision, recovered extractor and optional training setup.
Recovered working-tree files can differ from recorded Git HEAD; file hashes
and adaptation notices identify what is actually included. The source package
supports new, explicitly named runs without requiring a private RC_RL checkout.

The recovered EfficientZero extractor requests **15 named hooks**, while the
historical summary uses **11 numeric layer labels**. That mapping has not been
recovered. Likewise, the saved DDQN channels cannot automatically be identified
with every historical result-layer label. New baseline ROI rows require an
explicit layer map; the code does not relabel candidates as final paper inputs.

Available checkpoint/configuration candidates come from different generations.
Some actual EfficientZero training configurations use RGB observations and a
13-level curriculum, differing from the manuscript's baseline description.
Older candidate checkpoint paths were not all retained. Successful extraction
with an available checkpoint establishes code compatibility, not final-paper
checkpoint identity or an exact reproduction of the reported baseline.

## Coverage and analysis conventions

The separate audit found missing headline result cells and missing raw feature
tensors despite later aligned/results coverage. These gaps remain explicit;
absence is never encoded as zero performance. A temporal-shuffle caption and
the retained model/layer labels also need reconciliation with the original
figure-producing inputs. The code consolidation changes no historical scores.

Human play outcomes are nullable and death events must be checked before
classifying incompletes. Practice runs, cohort-specific upper levels and
participant-versus-model timeout differences are retained in the documented
comparison rules. Humans advanced on a fixed scanner schedule; model agents
used blocked advancement. Keypress counts, model decisions and engine frames
are different clocks. EMPA1 is the reported comparison; EMPA2 is excluded by
default. See the [reproduction guide](../reproducibility.md) and
[neural conventions](../../analysis/tomov23/docs/scientific-conventions.md).

Neural outputs distinguish feature variant/stream, performance band and fit
condition. Main-only and nuisance-adjusted fits must not be silently combined.
The original best-layer selection and stochastic fitting conventions are
preserved; restoring runnable code is not an independent validation of the
paper's scientific conclusions.

## What was exercised

The [dated validation summary](validation-2026-09-27.json) records the combined
source tests and representative runs. Checks include recorded human replay,
three-stream extraction through a tiny synthetic CPU model, baseline runtime
checks, and synthetic NIfTI-to-ROI integration. The tiny model validates the
extraction interface and timestamps; its features are not scientific paper
features. Historical figure commands were also exercised against separately
held tables; those tables are not embedded here.

Full participant fMRIPrep, whole-cohort fits, large-model GPU/distributed
extraction, full EfficientZero training and complete dataset upload were not
rerun. The [publication plan](huggingface-plan.md) keeps dataset staging and
public verification distinct from this source release.
