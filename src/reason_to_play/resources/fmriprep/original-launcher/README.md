# Original fMRIPrep launcher

This directory preserves the October 2025 RC_RL launcher and its
[upstream README](README.upstream.md) byte for byte. Source revision and
checksums are in [source.json](source.json).

For installation and execution, use the
[current preprocessing guide](../../../../../docs/guides/fmri-preprocessing.md).
The upstream README describes an old local setup: its participant exclusions,
29-participant total, home-directory paths and loader example are not the
current workflow. The study includes 32 participants.

The archived launcher invokes Docker; retained execution configurations report
Singularity. The exact producing image digest is unknown. See
[preprocessing sources](../../../../../docs/sources/README.md#fmri-preprocessing-sources)
for the preprocessing settings and evidence.
