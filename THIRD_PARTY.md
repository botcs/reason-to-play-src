# Source and license notices

This is a mixed-source research release. A root MIT notice does not override
component notices or supply missing permission for inherited source.

| Component | Source and license |
| --- | --- |
| Original Reason to Play code, documentation and newly created research artifacts | MIT, [LICENSE](LICENSE) |
| Inherited Colas harness/VGDL source | Development fork began from private `ccolas/infer-vgdl` on 11 January 2026. The related public [`ccolas/language_and_experience`](https://github.com/ccolas/language_and_experience/tree/5fa356d7b14ab2620cdbc38be7eb7eb699b63d0c) includes the preserved [MIT notice](licenses/colas-MIT.txt). The later public release is not the January fork point. |
| DeepSeek V3.2/V4 inference runtime | Upstream MIT notices in [deepseek_inference/LICENSE](deepseek_inference/LICENSE) and [deepseek_v4_inference/LICENSE](deepseek_v4_inference/LICENSE); exact source revisions in [sources](docs/sources/README.md#sources-and-implementations). Weights have separate terms. |
| EfficientZero inference modules | GPL-3.0 upstream; retain `baselines/vendor/efficientzero/LICENSE` and the file/hash/change ledger in its `PROVENANCE.json` |
| EfficientZero extraction scripts | Source snapshot from `A-Andrews/VGDL_FMRI_Comparisons`; no license file accompanies that snapshot. `baselines/extraction/efficientzero/PROVENANCE.json` records source bytes, attribution and adaptations. Included for this project's reproduction; no new third-party license grant is asserted. |
| RC_RL training/environment and original EMPA source | Three RC_RL runtime roles are included under `baselines/vendor/rc_rl/`; full-source pins and the external EMPA reference are in `baselines/sources.json`. Preserve inherited component terms and provenance; the MIT grant covers original project contributions. |
| Curated fMRI/alignment/encoding source | Original project contributions: MIT. Exact source mapping is in `docs/sources/code-origins.json`. Reported methodological influence from the Nature 2024 `tsumers/bert-brains` code is attributed in the source ledger; direct file descent has not been established. Any inherited third-party material retains its own terms. |

Original project code and newly created research artifacts use MIT. Raw
OpenNeuro ds004323 v1.0.0 records retain CC0. Third-party game assets,
checkpoints and model weights retain their component terms.

The V4 runtime's `model.py` and `encoding_dsv4.py` match its pinned official
source. Local adaptations retain the `out_dtype` quantization cast in
`kernel.py`, chunked checkpoint conversion/loading, and local encoder imports
in `generate.py`. Compare runtime revisions using the per-file source ledgers;
source agreement does not establish full-model distributed inference parity.

Datasets, model weights and participant catalogues are distributed
separately from this code repository. See the
[dataset card](docs/release/huggingface-dataset-card.md) for dataset contents
and licensing.
