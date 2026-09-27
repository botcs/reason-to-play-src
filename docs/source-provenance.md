# Source provenance

This release extends `botcs/reason-to-play-src` at
`309e9a7` (the original public snapshot). The development source consulted was
`botcs/llm-vgdl` at `3c441a925024c334005e50a784988ec5b16f642e`.
Local data, credentials and experiment-database contents are excluded. The optional experiment-database client is included with tracking disabled by default.

The fMRI/encoding snapshot is vendored from `botcs/tomov23-analysis` at
`a209c8441a950cada8812981fad8224b47c5ad87`, then reconciled with the original RC_RL launcher and the March/July analysis histories. The per-file [source map](../analysis/tomov23/docs/source-map.json) identifies those distinct inputs. The reconciled workflow was merged upstream in
[26d044f5](https://github.com/botcs/tomov23-analysis/commit/26d044f5a95ae60aa22c58f22fab3e3cb6363bea).
Its archive provenance and recovered fMRIPrep configurations are retained in `analysis/tomov23/`; the source does
not depend on an inaccessible Git submodule.

| Public files | Origin and changes |
| --- | --- |
| `analysis/tomov23/` | Curated preprocessing, BSON preparation, base/LLM alignment, encoder, ROI parser, and original fMRIPrep provenance from the pinned snapshot |
| `conf/extract_features/default.yaml` | Restored from the development commit; private tracking is disabled without embedding private bucket settings |
| `scripts/analysis/plot_behavioural.py` | `scripts/neurips_submission/plot_behavioural.py`; CSV and output paths are explicit, TeX optional, figure math retained |
| `scripts/analysis/plot_encoding.py` | `scripts/neurips_submission/plot_encoding.py`; portable paths and optional TeX |
| `scripts/analysis/build_episodes.py` | New offline export; uses the documented BSON/replay schema and the established outcome/death checks |
| `tests/test_{advancement_strategies,prompt_utils_schema,replay_codec,replay_harness,sliding_window_extraction,zstate_adapter}.py` | Existing regression tests restored from the development commit |
| `deepseek_inference/` | Tracked V3.2 runtime from the development commit, with upstream MIT notice retained |
| `deepseek_v4_inference/` | Tracked V4 runtime from the development commit; verified against the official Hugging Face snapshot and accompanied by its MIT notice and provenance |

DeepSeek V3.2 upstream code and MIT notice were verified against
[DeepSeek-V3.2 revision a7e62ac04ecb2c0a54d736dc46601c5606cf10a6](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6).
Local changes include batch/sequence defaults, fp32 score computation, output
NaN handling, and TileLang kernel adjustments.
The copied runtime includes local feature-extraction adaptations; it is not
claimed to be byte-identical to upstream HEAD. Model weights are separate and
must be obtained under their respective terms.

The V4 runtime license was recovered from the official
[DeepSeek-V4-Flash snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1);
the research `model.py` and `generate.py` matched that upstream revision.
The restored generator has small local import/chunk-loader adaptations; the
runtime README records these packaging changes and other local modifications.
The root MIT notice and the public Colas upstream notice are now included.
The development fork predates Colas's public release: its January source was
private `ccolas/infer-vgdl`, not a fork created from the later public repository.
[THIRD_PARTY.md](../THIRD_PARTY.md) records component license boundaries,
including GPL EfficientZero modules and recovered scripts without a found
license. The author selected MIT jointly for original project code and newly created research artifacts. Third-party source and model weights retain their own terms.

Baseline training sources, recovered extractor bytes, checkpoint maps and local
CPU-inference adaptations are recorded under [baselines/](../baselines/README.md).
Recovered working-tree files may differ from their repository HEAD; their
SHA-256 hashes are the evidence for the source actually recovered.

The [scientific provenance audit](release/scientific-provenance.md) distinguishes
runnable restored code from unresolved mappings to published results. In
particular, historical EfficientZero result-layer labels cannot be inferred
from the recovered extractor's hook list.
