#!/bin/bash
# Fine-tune on the CodeQA training dataset (chunked, stage-2 style).
#
# Prerequisites:
#   Build the compact parquet if not already done:
#     uv run python data/build_codeqa_compact.py
#
# Usage:
#   RUN_NAME=<stage1_run_dir> bash scripts/main_exp/3-train-codeqa-chunk.sh
#
# RUN_NAME should point to a directory under train_outputs/runs/ that contains
# a checkpoint-80000 from stage 1 (1-train.sh).

port=29051

uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
--num_processes=8 --gpu_ids all train.py \
configs/main_exp/codeqa_compact_chunk_l2l.yaml \
--model_name_or_path=google/gemma-2-2b-it \
--target_modules=down_proj \
--lora_r=8 \
--eval_strategy=no \
--max_qas_len=512 \
--max_qas_per_sample=1 \
--per_rank_gen=True \
--per_layer_processing=True \
--gen_lora_l1_reg_coef=0.1 \
--max_steps=20000 \
--gradient_accumulation_steps=16 \
--max_packed_inp_len=1024 \
--max_packed_ctx_len=2048 \
--use_per_ctx_average_loss=True \
--use_kl_loss=True \
--quantize_ctx_encoder=True \
--torch_empty_cache_steps=10 \
--from_pretrained_checkpoint=train_outputs/runs/$RUN_NAME/checkpoint-80000/pytorch_model.bin \
--max_ctx_chunk_len=512 \
--min_ctx_chunk_len=25 \
--num_chunk_probs='{"1":"0.5", "2":"0.125", "3":"0.0625", "4":"0.0625", "5":"0.0625", "6":"0.0625", "7":"0.0625", "8":"0.0625"}' \
--warmup_steps=2000 \
--learning_rate=2e-5
