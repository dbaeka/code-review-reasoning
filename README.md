# Thinking Before Generating Code Review Comment

## Description

This repository contains code for running inference and evaluating the performance of reasoning models on Code Review
Comment Generation using LLMs.

## Prerequisites

- Python 3.11
- vLLM
- PyTorch
- Transformers
- Datasets

## Installation

Install miniconda using the following command:

```bash
#!/bin/bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Set up Conda Environment

```bash
conda create --name inf_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate inf_env
```

### Clone the Repository

```bash
git clone https://github.com/dbaeka/code-review-reasoning
cd code-review-reasoning
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Inference

```bash
python inference.py --model_name {MODEL_NAME \
    --pause_duration {PAUSE_DURATION} --num_of_few_shot {NUM_OF_FEW_SHOT} --with_summary {WITH_SUMMARY} --with_callgraph {WITH_CALLGRAPH} \
    --num_of_results {NUM_OF_RESULTS} --seed {SEED} --batch_size {BATCH_SIZE} --temperature {TEMPERATURE} \
    --is_reasoning_model {IS_REASONING_MODEL} --base_drive_dir {BASE_OUTPUT_DIR} \
    --test_name {TEST_NAME} --total_shards {TOTAL_SHARDS} --provider {PROVIDER}
```

### Explanation of Arguments

- `model_name`: The name of the model to use for inference. Example: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `pause_duration`: The duration to pause between each generated code snippet. Default: `0.5`
- `num_of_few_shot`: The number of few-shot examples to use for inference. Default: `5`
- `with_summary`: Whether to include the summary in the prompt. Default: `False`
- `with_callgraph`: Whether to include the callgraph in the prompt. Default: `False`
- `num_of_results`: The number of results to generate per prompt. Default: `5`
- `seed`: The seed to use for generating the results. Default: `0`
- `batch_size`: The batch size to use for inference. Number of prompts to send at a go to model. Default: `32`
- `temperature`: The temperature to use for sampling. Default: `0.7`
- `is_reasoning_model`: Whether the model is a reasoning model. Default: `False`
- `base_drive_dir`: The base directory to save the results and log files. Default: `/home/ubuntu/soen691`
- `shard_size`: The size of the shard to use for saving the results and splitting the prompts. Default: `32`
- `test_name`: The name of the test set to run. Can be: `test_5000` or `test_500`
- `provider`: The provider to use for running the inference. Default: `vllm`. Other options are `groq`, `unsloth`
  and `openai`

## Copying the Results

```bash
rsync -avz -e ssh {USERNAME}@{SERVER_IP}:/home/ubuntu/soen691/ {DESTINATION_DIR_ON_LOCAL}
````

## Running Tests

To run the tests with pytest, run the following command:

```bash
pytest
```

## Repo Structure

- `inference.py`: The script for running inference on the models.
- `requirements.txt`: The file containing the dependencies for the project.
- `README.md`: The file containing the instructions for running the code.
- `tests/`: The directory containing the tests for the project.
- `misc/`: The directory containing the miscellaneous files for the project including replication material from Haider
  et al. (2022).
- `notebooks/`: The directory containing the notebooks for the project.
- `src/`: The directory containing the source code for the project including shred tools and service providers for
  running inference.
- `results/`: The directory containing the prompts and responses used per model. Also contains evaluation results from
  log output.
- `build_few_shot_examples.py`: The script to build the few shot examples and push to HF.
- `extract_bm_25_indices.py`: The script to build a temp of bm25 5 shot examples for quick referencing during few shot
  prompt generation.
- `build_zero_shot_examples.py`: The script to build the zero shot examples and push to HF.
- `create_500_dataset.py`: The script to build the 500 subset data used in this research. Data pushed to HF.
- `budget_force_inference.py`: The script for running inference on local model with budget forcing.
- `combine_results.py`: The script to put all results from results/output like data into a single HF dataset. Contains 5
  attempts per prompt results.
- `build_results.py`: The script to select the best result per prompt based on BLEU score. Pushed to HF.
- `evaluate_results.py`: The script to run BERTScore and BLEU evaluations against truth set.

## Evaluation

Evaluation is done using BLEU, BERTScore and ExactMatch. Only BLEU and BERTScore are reported for this project. Empty
lines are ignored in the final results selected. See `results/evaluation_results.txt` for detailed results per model and
prompt type.

## HuggingFace Copies

### Datasets

- [Few Shot (Without Semantic Metadata)](https://huggingface.co/datasets/dbaeka/soen_691_few_shot_test_500_base_hashed)
- [Few Shot (With Summary Semantic Metadata)](https://huggingface.co/datasets/dbaeka/soen_691_few_shot_test_500_summary_hashed)
- [Few Shot (With Summary + Callgraph Semantic Metadata)](https://huggingface.co/datasets/dbaeka/soen_691_few_shot_test_500_summary_callgraph_hashed)
- [Few Shot (With Callgraph Semantic Metadata)](https://huggingface.co/datasets/dbaeka/soen_691_few_shot_test_500_callgraph_hashed)
- [Zero Shot](https://huggingface.co/datasets/dbaeka/soen_691_zero_shot_test_500_hashed)
- [Results (5 Attempts)](https://huggingface.co/datasets/dbaeka/soen_691_test_500_hashed_with_results)
- [Final Results (Best Response Chosen)](https://huggingface.co/datasets/dbaeka/soen_691_test_500_final_selected_results)

### Models

Open Sourced Models are cloned to preserve their states and weights to our repo, except gated models like Llama 3.1. Links are below. The original model
repo can be traced when opened on HuggingFace.

- [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/dbaeka/DeepSeek-R1-Distill-Llama-8B)