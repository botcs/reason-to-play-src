# DeepSeek V3.2 Feature Extraction with Torchrun

Extract hidden states from DeepSeek V3.2 using tensor parallelism.

## Prerequisites

1. DeepSeek inference code at `deepseek_inference/` (already in repo)
2. Converted model shards (see below)
3. Multi-GPU setup (8 GPUs for 671B model)

## Step 1: Convert HF Checkpoint to Sharded Format

```bash
cd deepseek_inference

python convert.py \
    --hf-ckpt-path /path/to/DeepSeek-V3-0324 \
    --save-path /path/to/output/shards \
    --n-experts 256 \
    --model-parallel 8
```

This creates files like:
- `model0-mp8.safetensors`
- `model1-mp8.safetensors`
- ...
- `model7-mp8.safetensors`

## Step 2: Run Feature Extraction

```bash
torchrun --nproc-per-node 8 -m src.llm_eval.human_replay.extract_features \
    --model deepseek-ai/DeepSeek-V3-0324 \
    --deepseek-ckpt-path ~/.cache/torchrun/deepseekv32 \
    --deepseek-config deepseek_inference/config_671B_v3.2.json \
    --prompts out/prompts/*.jsonl \
    --hf-batch-size 4 \
    --output-dir out/human_replay_features
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID (used for model_id in output paths) |
| `--deepseek-ckpt-path` | Path to converted shards directory |
| `--deepseek-config` | Path to model config JSON |
| `--prompts` | Input prompt files (glob patterns supported) |
| `--hf-batch-size` | Batch size for extraction |
| `--output-dir` | Output directory for features |
| `--frame-stride` | Sample every Nth prompt (default: 1) |

## Output

Features saved to: `out/human_replay_features/model-{model_id}/{subject}/{game}/level_NN.pt`

Each `.pt` file contains:
- `hidden_states`: tensor of shape `(num_steps, num_layers+1, hidden_dim)`
- Metadata about the play session

## Notes

- All 8 GPUs must participate in every forward pass (tensor parallel collectives)
- Only rank 0 performs I/O operations
- The wrapper extracts hidden states at the last token position for each prompt
