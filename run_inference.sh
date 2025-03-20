BASE_DRIVE_DIR="/Users/dbaeka/Library/CloudStorage/GoogleDrive-dbaekajnr@gmail.com/My Drive/"
SHARD_SIZE=32
TEST_NAME="test_5000"
BATCH_SIZE=1
WITH_SUMMARY=0
WITH_CALLGRAPH=0
SEED=0
NUM_OF_RESULTS=5
NUM_OF_FEW_SHOT=5
TEMPERATURE=0.7
IS_REASONING_MODEL=1
PAUSE_DURATION=4
BATCH_CALL=0
MODEl_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TOTAL_SHARDS=157

python3 ./inference_groq.py - --model_name $MODEl_NAME \
                       --pause_duration $PAUSE_DURATION \
                       --num_of_few_shot $NUM_OF_FEW_SHOT \
                       --with_summary $WITH_SUMMARY \
                       --with_callgraph $WITH_CALLGRAPH \
                       --num_of_results $NUM_OF_RESULTS \
                       --seed $SEED \
                        --batch_size $BATCH_SIZE \
                        --temperature $TEMPERATURE \
                        --is_reasoning_model $IS_REASONING_MODEL \
                        --batch_call $BATCH_CALL \
                        --base_drive_dir "$BASE_DRIVE_DIR" \
                        --shard_size $SHARD_SIZE \
                        --test_name $TEST_NAME \
                        --total_shards $TOTAL_SHARDS \
                        --num_instances 2
