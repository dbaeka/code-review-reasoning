#!/bin/bash

# Constants
HOST="0.0.0.0"
BASE_PORT=11432
#VLLM_BINARY="/usr/local/bin/vllm"
VLLM_BINARY="/Users/dbaeka/miniconda3/envs/vllm/bin/vllm"
LOG_DIR="vllm-server-logs"

# Check if the number of GPUs is provided as an argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Command-line argument
MODEL=$1

# Check if the binary exists
if [[ ! -x "$VLLM_BINARY" ]]; then
    echo "Error: VLLM binary not found or not executable at $VLLM_BINARY"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start server instances
LOG_FILE="${LOG_DIR}/${PORT}.log"

# Start server with nohup and log output
#nohup "$VLLM_BINARY" serve "${MODEL}" --host ${HOST} --port ${BASE_PORT} --enable-reasoning --reasoning-parser deepseek_r1 --quantizatoin bitsandbytes --load-format bitsandbytes --dtype bfloat16 --gpu-memory-utilization 0.8 --tensor-parallel-size 1 > "$LOG_FILE" 2>&1 &

nohup "$VLLM_BINARY" serve "${MODEL}" --host ${HOST} --port ${BASE_PORT} --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 1 llm_model  > "$LOG_FILE" 2>&1 &


if [[ $? -eq 0 ]]; then
    echo "Started server instance on port ${BASE_PORT}, logging to ${LOG_FILE}"
else
    echo "Error: Failed to start server instance on port ${PORT}"
fi

