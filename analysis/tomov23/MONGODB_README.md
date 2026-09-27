# Behavioral BSON data used by the release

The raw source is OpenNeuro ds004323 version 1.0.0, in
`behavior/dump.tar.gz`, which expands to `dump/heroku_7lzprs54/`.
The original archive's `behavior/README` lists the collections. The scripts
read BSON with PyMongo directly; a MongoDB server is not required.

The older working-branch version of this document contained incomplete
statistics and incorrect outcome shortcuts. It remains in the archived Git
history but should not be used for the release's analysis.

## Files and identifiers

- `plays.bson`: gameplay episodes, including `subj_id`, `run_id`, game/level
  identity, compressed zstates, and the play-level `win` outcome.
- `runs.bson`: scanner run timing and metadata used for TR alignment.
- `regressors.bson`: EMPA-derived theory strings used for HRR representations.
- `stages/prepare_behavioral_data.py` partitions the original play documents
  into `plays/sub-NN/run-NN.bson` and copies `runs.bson` unchanged.
- `run-00` is out-of-scanner Sokoban practice and is excluded from comparisons.
  Preserve original BSON object IDs when joining play-level features.

## Verified analysis conventions

These were checked against the source data in the research repository on
2026-07-29; consult the release's behavioral analysis scripts for the executable
classification rules.

- **Use the play document's `win` field.** It is three-valued: `True` is a win,
  `False` means a loss rule fired, and `None` can be an incomplete recording.
  Avatar deaths also have `win=None` and `ended=False`; detect the avatar's
  `killSprite` event before classifying remaining `None` plays as incomplete.
  The terminal zstate's `win=-1` is a placeholder for `None`, not a valid loss
  label. Preserve the original three values when exporting tables.
- **The zstate index is the engine tick.** `states[i]['gt'] == i`, so a play
  with N states has a terminal engine clock of N-1. An LLM decision is a
  separate clock and may advance several engine frames.
- **Respect the cohort.** sub-01 through sub-11 played vgfmri3 (levels 0-11);
  sub-12 through sub-32 played vgfmri4 (levels 0-8). Restrict cross-agent level
  comparisons to levels 0-8. Scan availability should be discovered from input
  files: sub-09 has five runs in the recovered fMRIPrep records.
- **Timeouts differ by source game.** Original vgfmri4 bait/chase/helper/
  lemmings/zelda had no Timeout rule; vgfmri4 avoidGeorge awarded a timed
  survival win. All vgfmri3 games had a losing 600-tick timeout. Converted
  simulator files can differ from these original human-play rules.
- **Advancement differs between participants and model runs.** Participants
  advanced on a fixed 60-second schedule; the reported model curricula required
  consecutive wins. State the solve-rate denominator and advancement rule.
- **Model rationales are not environment ground truth.** Verify layouts and
  objects in the recorded observations. Level resets are deterministic.

For the full pipeline and input paths, see [README.md](README.md).
