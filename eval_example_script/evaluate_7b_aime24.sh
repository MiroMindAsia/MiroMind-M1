#!/bin/bash

# Define common variables
export HF_HOME=./cache/huggingface
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# If you need a proxy, you can set it here
#export HF_ENDPOINT=https://hf-mirror.com

NUM_GPU=4

MODEL_PATHS=binwang/MiroMind-M1-RL-7B
MODEL_NAME=MiroMind-M1-RL-7B
DATASET=aime24
NUM_RUNS=1

echo "========================================="
echo "Running evaluation"
echo "DATASET: $DATASET"
echo "NUM_RUNS: $NUM_RUNS" 
echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_PATHS: $MODEL_PATHS"
echo "========================================="

cd m3eval

python main.py \
    --model_path "$MODEL_PATHS" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --tensor_parallel_size $NUM_GPU \
    --default_sys_msg False \
    --reasoning_in_sys_msg False \
    --batch_multiple_runs True \
    --enforce_eager True \
    --enable_chunked_prefill True \
    --ignore_gpqa_instruction True \
    --temperature 0.6 \
    --top_p 0.95 \
    --num_runs "$NUM_RUNS" \
    --max_tokens 64000 \
    --output_dir ../results/

echo "========================================="


