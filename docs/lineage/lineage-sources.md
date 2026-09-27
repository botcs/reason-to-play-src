# About the lineage figures

Prepared on 27 September 2026 as an unpublished history review. These are
standalone explanatory artifacts; producing them did not publish a site,
change repository history. Subsequent consolidation is documented in the current reproduction guide.

- **lineage-all.pdf** contains both pages and is the preferred shareable copy.
- **lineage.png / .svg / .pdf** show the project overview.
- **lineage-analysis-branches.png / .svg / .pdf** show exact analysis ancestry.
- **generate_lineage.py** regenerates every figure with Python and matplotlib.
  PNGs are 160 dpi; SVG/PDF text remains selectable. Cards link to primary
  evidence in the vector exports. Private repository links require access.

Dates are milestones, not a metric time axis. Solid arrows represent verified
Git ancestry and may omit intervening commits. Dotted arrows represent a
verified data, content or dependency relationship; they do not establish a
Git-parent relationship. The dashed Nature-to-VGDL edge represents adaptation
reported by the project owner. Shared author and methods are independently
verified, but exact source-file copying has not been established. Recruitment
and paper acceptance dates are author-reported rather than inferred from Git.

The diagram deliberately keeps the later public Colas snapshot separate from
the private source used in January. The February snapshot's commit date does
not establish the precise moment that the repository became public.

## Evidence index

The companion [TIMELINE.md](TIMELINE.md) contains the dated cross-repository
record. [tomov23-analysis.md](tomov23-analysis.md) provides the parent and blob
comparisons, [nature-source.md](nature-source.md) identifies the earlier paper
and official source, and [baselines.md](baselines.md) documents EfficientZero
and DDQN provenance.

| Figure element | Primary evidence |
| --- | --- |
| Original experiment | [tsividis/vgdl reference branch](https://github.com/tsividis/vgdl/tree/refactor_fMRI_cannon); [RC_RL reference correction](https://github.com/botcs/RC_RL/commit/15586a43fe054cdca89edab46fd8efef91228a9c) |
| Raw human fMRI | [OpenNeuro ds004323 v1.0.0](https://openneuro.org/datasets/ds004323/versions/1.0.0). The figure does not claim the study's MongoDB BSON dump is hosted in this dataset. |
| RL fork ancestry | [tomov/RC_RL upstream base 8df431a](https://github.com/tomov/RC_RL/commit/8df431a1b5311ad8ff9d1a55e0a3a04076128ab3); [first botcs work 99566fd](https://github.com/botcs/RC_RL/commit/99566fd07cb5d31da16802b2b2422109af289441) |
| Raw preprocessing | [Original fMRIPrep launcher](https://github.com/botcs/RC_RL/blob/969141d09369ecd783965fadbbe2a1920065cf24/fmriprep_pipeline/run_fmriprep.sh), first committed 16 October 2025; retained execution timestamps are catalogued in [fmriprep-invocations.csv](fmriprep-invocations.csv). |
| EfficientZero collaborator fork | [A-Andrews/EfficientZeroV2 retained Jan 17 revision](https://github.com/A-Andrews/EfficientZeroV2/tree/29157d4892afd9467b1bd0994de1355086145490), fork of [Shengjiewang-Jason/EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2). The VGDL wrapper imports `RC_RL.VGDLEnv`; that is a code dependency, not Git ancestry from RC_RL. |
| EfficientZero milestones | [VGDL, Oct 6](https://github.com/A-Andrews/EfficientZeroV2/commit/09e62d2cd059eb1e78f3e50deac0ae8984aa764c); [integrated wrapper, Oct 29](https://github.com/A-Andrews/EfficientZeroV2/commit/bf13e302dd3f72c6bbaa7b15ae9354eccc89c482); [curriculum, Nov 13](https://github.com/A-Andrews/EfficientZeroV2/commit/8597088064cb5b9e0f3d6f366d87d850e306bd92). Contributor is identified as Austin, not inferred from the username. |
| Earlier neural methodology | [Kumar, Sumers et al., Nature Communications 15, 5523 (2024)](https://www.nature.com/articles/s41467-024-49173-5); [first ridge-analysis commit, 20 Apr 2022](https://github.com/tsumers/bert-brains/commit/0574549cfdda381277d20b5c82e09e6da83042ca); [archived official code](https://doi.org/10.5281/zenodo.10863840). This work used Narratives ds002345, not ds004323. |
| First encoding import | [3199db9, 17 Dec 2025](https://github.com/botcs/tomov23-analysis/commit/3199db9946ada6328fab419d2c5f9fb1bb9c5ca3), parent `d0fc2a0d459b5f796ce75fe9582bf0a09f0e99d3`. |
| Parcel-to-voxel branch | [3e0f53a, 28 Jan 2026](https://github.com/botcs/tomov23-analysis/commit/3e0f53a750c6544fa9a6261adeb04c6441f08803), via `718e09cfa693908940ed5711447dc5936a2ccb77`. |
| Two sibling imports | [26fa74c, 23 Mar 2026](https://github.com/botcs/tomov23-analysis/commit/26fa74ce715781bfdd7a6f6a4bc2a6685e5a8678) and [62c2021, 31 Jul 2026](https://github.com/botcs/tomov23-analysis/commit/62c202130d057e585b6bf7978f621fcd0255919c) both have immediate parent `d8eaa5a6c7fc3382967cb9e004748c89c99eb526`. Their preprocessing file is the same Git blob `3f78dbfa617cca456b0e9301a879e93820a7ea77`; the aligners and encoders differ. |
| January LLM fork boundary | First botcs commit [49a2bca](https://github.com/botcs/llm-vgdl/commit/49a2bca26ccd03aca5d3d67be9b6d2f3fb71b11c) has parent `64dc4f6d4621acdf289819f198ba1ca318b621dd`, verified against private `ccolas/infer-vgdl`; [first harness 68fc1024](https://github.com/botcs/llm-vgdl/commit/68fc1024013508633d0eeb00780ae6d5a3cbe74a). |
| Later Colas public snapshot | [edb30bc, 12 Feb 2026](https://github.com/ccolas/language_and_experience/commit/edb30bc9815efad268605d337d63601d4f6f39a8). |
| Public artifacts | [Initial source export, 9 May](https://github.com/botcs/reason-to-play-src/commit/9e5b2c8); [website creation, 7 May](https://github.com/botcs/reason-to-play/commit/c4583a2); [website link to separate source, 9 May](https://github.com/botcs/reason-to-play/commit/e20c204). |

The figures describe sources and preserved history. They do not certify that
any two historical pipelines produce equivalent results, or identify the
producing revision of every current S3 object.
