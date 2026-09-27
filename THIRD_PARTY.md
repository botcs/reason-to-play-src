# Source and license notices

This is a mixed-source research release. A root MIT notice does not override
component notices or supply missing permission for inherited source.

| Component | Source and license |
| --- | --- |
| Original Reason to Play code, documentation and newly created research artifacts | MIT, [LICENSE](LICENSE) |
| Inherited Colas harness/VGDL source | Development fork began from private `ccolas/infer-vgdl` on 11 January 2026. The related public [`ccolas/language_and_experience`](https://github.com/ccolas/language_and_experience/tree/5fa356d7b14ab2620cdbc38be7eb7eb699b63d0c) includes the preserved [MIT notice](licenses/colas-MIT.txt). The later public release is not the January fork point. |
| DeepSeek V3.2/V4 inference runtime | Upstream MIT notices and exact source revisions inside `deepseek_inference/` and `deepseek_v4_inference/`; weights have separate terms |
| EfficientZero inference modules | GPL-3.0 upstream; retain `baselines/vendor/efficientzero/LICENSE` and the file/hash/change ledger in its `PROVENANCE.json` |
| Recovered EfficientZero extraction scripts | S3 working-tree recovery from `A-Andrews/VGDL_FMRI_Comparisons`; no license file was present in that backup. `baselines/recovered/efficientzero/PROVENANCE.json` records source bytes, attribution and adaptations. Included for this project's reproduction; no new third-party license grant is asserted. |
| RC_RL training/environment and original EMPA source | Three audited RC_RL runtime roles are included under `baselines/vendor/rc_rl/`; full-source pins and the external EMPA reference are in `baselines/sources.json`. Preserve inherited component terms and provenance; the MIT grant covers original project contributions. |
| Curated fMRI/alignment/encoding source | Original project contributions: MIT. Exact source mapping is in `analysis/tomov23/docs/source-map.json`. Reported methodological influence from the Nature 2024 `tsumers/bert-brains` code is attributed in the source ledger; direct file descent has not been established. Any inherited third-party material retains its own terms. |

The author selected MIT jointly for original code and new research artifacts
on 27 September 2026. Raw OpenNeuro ds004323 v1.0.0 remains CC0; this decision
does not replace inherited terms for third-party game assets or checkpoints. Model repositories govern their
weights. No model weights, raw MRI, BSON payloads, participant catalogues,
historical behavioral tables or detailed storage inventories are distributed
in this code-only Git release. Those materials belong to a separately reviewed
dataset release; a software license does not assert that a dataset is published.
