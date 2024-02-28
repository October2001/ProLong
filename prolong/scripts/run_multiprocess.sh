#!/bin/bash
set -ux

# hyper-parameters
MODEL_PATH=facebook/opt-350m
CHUNK_SIZE=128
WINDOW_SIZE=32768
SINGLE_PPL_BATCH_SIZE=256
PAIR_PPL_BATCH_SIZE=256
SAMPLE_SIZE=500
SEED=11
DLT_PPL_THRESHOLD=0.1

# output settings
DATA_FILE_PATH=<YOUR_DATA_PATH>
ROOT_PATH=<YOUR_OUTPUT_ROOT_PATH>

python run_batch_multi_process.py \
    --data_file $DATA_FILE_PATH \
    --root_path $SAVE_FILE \
    --model_name $MODEL_PATH \
    --use_flash_attention_2 \
    --chunk_size $CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    --single_ppl_batch_size $SINGLE_PPL_BATCH_SIZE \
    --pair_ppl_batch_size $PAIR_PPL_BATCH_SIZE \
    --sample_size $SAMPLE_SIZE \
    --need_draw \
    --seed $SEED \
    --dlt_ppl_threshold $DLT_PPL_THRESHOLD \
    --gpu_ids 0 1 2 3

    