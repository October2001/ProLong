#!/bin/bash
set -ux

INPUT_DIR=<YOUR_INPUT_DIR>
FILE_LIST=filelist # this will combine with input dir to get the full path of each file, eg. <YOUR_INPUT_DIR>/file0.jsonl
OUTPUT_DIR=<YOUR_OUTPUT_DIR>
IP_HOSTFILE=iphost
MODEL_PATH=<YOUR_MODEL_PATH>
SAMPLE_SIZE=500
CHUNK_SIZE=128
WINDOW_SIZE=32768
SINGLE_PPL_BATCH_SIZE=256
PAIR_PPL_BATCH_SIZE=256
SEED=11
DLT_PPL_THRESHOLD=0.1

python run_batch_multinodes.py \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --ip-hostfile $IP_HOSTFILE \
    --file-list $FILE_LIST \
    --model_name $MODEL_PATH \
    --chunk_size $CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    --single_ppl_batch_size $SINGLE_PPL_BATCH_SIZE \
    --pair_ppl_batch_size $PAIR_PPL_BATCH_SIZE \
    --sample_size $SAMPLE_SIZE \
    --use_flash_attention_2 \
    --need_draw \
    --seed $SEED \
    --dlt_ppl_threshold $DLT_PPL_THRESHOLD