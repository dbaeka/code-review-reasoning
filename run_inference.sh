BASE_DRIVE_DIR="/home/ubuntu"
SHARD_SIZE=2048
TEST_NAME="test_500"
BATCH_SIZE=2048
WITH_SUMMARY=0
WITH_CALLGRAPH=0
SEED=0
NUM_OF_RESULTS=5
NUM_OF_FEW_SHOT=5
TEMPERATURE=0.7
IS_REASONING_MODEL=1
PAUSE_DURATION=4
MODEl_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

python inference.py --model_name $MODEl_NAME \
                    --pause_duration $PAUSE_DURATION \
                    --num_of_few_shot $NUM_OF_FEW_SHOT \
                    --with_summary $WITH_SUMMARY \
                    --with_callgraph $WITH_CALLGRAPH \
                    --num_of_results $NUM_OF_RESULTS \
                    --seed $SEED \
                    --batch_size $BATCH_SIZE \
                    --temperature $TEMPERATURE \
                    --is_reasoning_model $IS_REASONING_MODEL \
                    --base_drive_dir $BASE_DRIVE_DIR \
                    --shard_size $SHARD_SIZE \
                    --test_name $TEST_NAME  \
                    --provider vllm
