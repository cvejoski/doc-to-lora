#!/bin/bash
# Evaluate on the vm2825/CodeQA-dataset dataset.
#
# Usage examples:
#   # Trained hypernet checkpoint (Doc-to-LoRA):
#   RUN_NAME=<run_dir> step=<step> bash scripts/main_exp/eval/codeqa.sh
#
#   # Base model (no context, upper-bound baseline):
#   MODEL=google/gemma-2-2b-it bash scripts/main_exp/eval/codeqa.sh
#
# Environment variables:
#   RUN_NAME  – directory under train_outputs/runs/ that contains args.yaml
#   step      – checkpoint step number (e.g. 20000)
#   MODEL     – HuggingFace model ID to evaluate as a plain base model
#   SPLIT     – dataset split to evaluate on (default: test)
#   BATCH_SIZE – generation batch size (default: 32)
#   MAX_CTX_CHUNK_LEN – maximum context chunk length passed to the hypernet
#                       (default: 512, set to 8192 for full-context eval)

SPLIT="${SPLIT:-test}"
MAX_CTX_CHUNK_LEN="${MAX_CTX_CHUNK_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [ -n "$RUN_NAME" ] && [ -n "$step" ]; then
    # ------------------------------------------------------------------
    # Doc-to-LoRA hypernet evaluation (query internalization mode)
    # ------------------------------------------------------------------
    CHECKPOINT="train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin"

    echo "=== D2L hypernet: internalize code context then answer ==="
    WANDB_MODE=disabled uv run run_eval.py \
        --checkpoint_path "$CHECKPOINT" \
        --datasets codeqa \
        --split "$SPLIT" \
        --max_ctx_chunk_len "$MAX_CTX_CHUNK_LEN" \
        --eval_batch_size_gen "$BATCH_SIZE"

elif [ -n "$MODEL" ]; then
    # ------------------------------------------------------------------
    # Base-model evaluation (context prepended to input)
    # ------------------------------------------------------------------
    echo "=== Base model with context in input ==="
    WANDB_MODE=disabled uv run run_eval.py \
        --model_name_or_path "$MODEL" \
        --datasets codeqa \
        --split "$SPLIT" \
        --add_ctx_to_input \
        --truncate_if_too_long_inp \
        --eval_batch_size_gen "$BATCH_SIZE"

else
    echo "Error: set either (RUN_NAME + step) for a hypernet checkpoint"
    echo "       or MODEL for a plain base-model evaluation."
    exit 1
fi
