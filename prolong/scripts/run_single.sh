DATA_FILE_PATH=<YOUR_DATA_PATH>
SAVE_PATH=<YOUR_SAVE_PATH>
SCORE_PATH=<YOUR_SCORE_PATH>
PIC_DIR=<YOUR_PIC_DIR>
CHUNK_SIZE=128
DLT_PPL_THRESHOLD=0.1
WINDOW_SIZE=32768
MODEL_PATH=facebook/opt-350m
SEED=11
SINGLE_PPL_BATH_SIZE=256
PAIR_PPL_BATH_SIZE=256
SAMPLE_SIZE=500

python run.py \
    --data_file $DATA_FILE_PATH \
    --save_file $SAVE_PATH \
    --score_file $SCORE_PATH \
    --pic_dir $PIC_DIR \
    --chunk_size $CHUNK_SIZE \
    --dlt_ppl_threshold $DLT_PPL_THRESHOLD \
    --window_size $WINDOW_SIZE \
    --model_name $MODEL_PATH \
    --seed $SEED \
    --single_ppl_batch_size $SINGLE_PPL_BATH_SIZE \
    --pair_ppl_batch_size $PAIR_PPL_BATH_SIZE \
    --sample_size $SAMPLE_SIZE