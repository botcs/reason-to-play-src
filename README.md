# Reason to Play

Source code for **Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners.**

_Botos Csaba, Sreejan Kumar, Austin Tudor David Andrews, Laurence Hunt, Chris Summerfield, Joshua B. Tenenbaum Rui Ponte Costa, Marcelo G. Mattar, Momchil Tomov_

Live site, interactive replay catalogue, representation viewer, and paper:
**[botcs.github.io/reason-to-play](https://botcs.github.io/reason-to-play/)**

> **Acknowledgement:** This project is a fork of, and
> directly inspired by, Cedric Colas's
> [Language and Experience: A Computational Model of Social Learning in
> Complex Tasks](https://github.com/ccolas/language_and_experience).
> The VGDL harness, game suite, and some of the LLM scaffolding originate
> there; we extend the codebase to study frontier Large Reasoning Models
> learning these games without rule descriptions, and to align their
> internal representations against human fMRI data. Huge thanks to Cedric
> and collaborators for granting us early access to the codebase.

We drop human participants (32, scanned with fMRI) and a suite of frontier Large Reasoning Models into a set of simple grid-world video games written in a compact language called VGDL, with no rules provided, and ask whether the models learn the way humans do, and whether they build similar internal representations.

## Setup

Minimal install (OpenRouter-backed gameplay and action-only human replay; no local GPU inference):

```bash
pip install -r requirements.txt
```

Full install (adds local LRM inference, Phase-2 hidden-state feature extraction, plotting, and S3 helpers). Torch is pinned to the CUDA 12.8 build, so pass the matching wheel index:

```bash
pip install -r requirements-full.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

Versions in both files are pinned to the working uv venv on our cluster (Python 3.12, snapshot 2026-05); loosen as needed for your environment.

We developed and optimised the local-inference path for a single **HGX B200 node (8 x B200 GPUs, NVLink)**. The same stack also runs on older NVIDIA architectures (Hopper / H100, Ampere / A100); install the torch / triton build that matches your CUDA stack and drop the Blackwell-tuned kernel extras (`tilelang`, `flash-linear-attention`, `causal-conv1d`, `torch-c-dlpack-ext`) from `requirements-full.txt` on hardware that does not need them.

## Quickstart

Generative gameplay (LRM plays a game from scratch, no rules):

```bash
python -m src.llm_eval.generative_gameplay.run \
    game.game=bait_vgfmri4 \
    llm.backend=openrouter llm.model=<MODEL> \
    harness.rationale_mode=copied-reasoning
```

Human replay imputation (run an LRM through a participant's recorded
keypresses to reconstruct the per-step "think aloud"):

```bash
python -m src.llm_eval.human_replay.run_replay \
    replay.subject=sub-01 \
    harness.rationale_mode=action-only
```

## Contents

```
src/
  llm_eval/                  LLM evaluation pipeline
    shared/                  config, harness, prompt loading, event logging,
                             observation formatting, response parsing,
                             LLM wrappers (OpenRouter, Transformers, torchrun)
    generative_gameplay/     LLM plays games (agent loop, Hydra entry point)
    human_replay/            replay pipeline (imputation + feature extraction)
  vgdl/                      VGDL game engine (interpreter, sprites, renderer)
prompts/
  gameplay/                  system prompts by rationale_mode x suggestion_level
  replay/                    imputation prompts for human replay pipeline
  game_rules/                per-game oracle rule descriptions
games/                       VGDL game description + level layout files
sweeps/                      W&B sweep configs for the experiment grid
conf/
  config.yaml                Hydra default configuration
figures/                     headline behavioural + neural figures
```

## Headline results

### Do LRMs learn the way humans do?

We compare how quickly each agent discovers the rules of a game, and how far through a curriculum of nine difficulty levels it can progress. Human participants, deep-RL baselines (DDQN, EfficientZero, EMPA), and eight frontier LRMs all play the same games under the same conditions.

The best LRMs cluster tightly around the human learning distribution. On the discovery metric, the top LRM is nearly indistinguishable from the human median; on the curriculum metric, it tracks human-level progression through all nine difficulty levels. The deep-RL baselines, by contrast, are far slower and plateau much earlier.

![Learning efficiency and capability](figures/behavioural_discovery_curriculum_combined.svg)

> **Learning efficiency and capability.** Top: discovery-time distributions (KDE); LRMs overlap with humans while deep-RL baselines are shifted far right. Bottom: curriculum progression under blocked advancement (two consecutive wins required to advance); the best LRMs track the human staircase closely.

### Do they build similar brain representations?

We extract hidden-state activations from each LRM during gameplay and use them as regressors in a voxelwise encoding model that predicts the human fMRI BOLD signal, with separate regularisation for the model features and the nuisance regressors (game / level identity, button presses, time). Best-layer Pearson correlations are then averaged within functional region groups.

LRM representations predict brain activity significantly above chance across visual, frontoparietal and default-mode regions, and outperform DDQN and EfficientZero baselines by a wide margin across every cortical region group. Targeted ablations (prompt-only, shuffled-features, random-init controls) confirm that the signal comes from the model's in-context representation of the game state, not from surface-level prompt statistics or chance correlations.

![Brain encoding accuracy by region group](figures/encoding_groups_combined.svg)

> **Brain encoding accuracy by region group.** Best-layer Pearson correlation averaged across voxels within each functional group. LRM features (right) consistently outperform deep-RL baselines (left) across cortical regions.

## Cite

```bibtex
@misc{csaba2026reasonplaybehavioralbrain,
      title={Reason to Play: Behavioral and Brain Alignment Between Frontier LRMs and Human Game Learners},
      author={Botos Csaba and Sreejan Kumar and Austin Tudor David Andrews and Laurence Hunt and Chris Summerfield and Joshua B. Tenenbaum and Rui Ponte Costa and Marcelo G. Mattar and Momchil Tomov},
      year={2026},
      eprint={2605.08019},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08019},
}
```

## License

MIT. See `LICENSE`.
