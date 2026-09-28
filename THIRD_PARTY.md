# Source and license notices

This is a mixed-source research release. A root MIT notice does not override
component notices or supply missing permission for inherited source.

| Component | Source and license |
| --- | --- |
| Original Reason to Play code, including collaborator contributions, documentation and newly created research artifacts | MIT, [LICENSE](LICENSE) |
| Inherited Colas harness/VGDL source | Development fork began from private `ccolas/infer-vgdl` on 11 January 2026. The related public [`ccolas/language_and_experience`](https://github.com/ccolas/language_and_experience/tree/5fa356d7b14ab2620cdbc38be7eb7eb699b63d0c) includes the preserved [MIT notice](licenses/colas-MIT.txt). The later public release is not the January fork point. |
| DeepSeek V3.2/V4 inference runtime | Upstream MIT notices in [agents/lrm/backends/deepseek_v32/LICENSE](agents/lrm/backends/deepseek_v32/LICENSE) and [agents/lrm/backends/deepseek_v4/LICENSE](agents/lrm/backends/deepseek_v4/LICENSE); [V3.2 source record](agents/lrm/backends/deepseek_v32/PROVENANCE.json) and [V4 source record](agents/lrm/backends/deepseek_v4/PROVENANCE.json) identify comparison revisions and per-file adaptations. Weights have separate terms. |
| EfficientZero inference modules | GPL-3.0 upstream; retain `agents/efficientzero/inference/LICENSE` and the file/hash/change ledger in its `PROVENANCE.json` |
| Optional EfficientZero training submodule | Austin Andrews's [EfficientZeroV2 fork](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490), GPL-3.0 upstream, pinned at `29157d4892afd9467b1bd0994de1355086145490` under `agents/efficientzero/training/`. Preserve its license and source notices. |
| RC_RL training/environment and original EMPA source | DDQN runtime roles are included under `agents/ddqn/`; EfficientZero's environment is under `agents/efficientzero/environment/`. Source pins are in `agents/ddqn/sources.json` and `agents/efficientzero/sources.json`; EMPA source references are in `agents/empa/sources.json`. Preserve inherited component terms and provenance; the MIT grant covers original project contributions. |
| Curated fMRI/alignment/encoding source | Original project contributions: MIT. Exact source mapping is in `docs/sources/code-origins.json`. Reported methodological influence from the Nature 2024 `tsumers/bert-brains` code is attributed in the source ledger; direct file descent has not been established. Any inherited third-party material retains its own terms. |

Original project code and newly created research artifacts use MIT. Raw
OpenNeuro ds004323 v1.0.0 records retain CC0. Third-party game assets,
checkpoints and model weights retain their component terms.

The runtime source records include upstream comparison revisions, current and
source hashes, and local adaptations. V4's `model.py` uses a package-relative
kernel import; `encoding_dsv4.py` has formatting and unused-name differences.
Other adaptations include the `out_dtype` quantization cast in `kernel.py`,
chunked checkpoint conversion/loading, and local encoder imports in `generate.py`.
These comparisons do not establish full-model distributed inference parity.

Datasets, model weights and participant catalogues are distributed
separately from this code repository. See the
[dataset card](docs/release/huggingface-dataset-card.md) for dataset contents
and licensing.
