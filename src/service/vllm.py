import logging
from time import sleep

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.service.common import extract_cot_and_answer, get_unprocessed_examples, save_results

vllm_model = None
tokenizer = None
max_seq_length = 4096

vllm_model_map = {
    # "Qwen/QwQ-32B": "qwen-qwq-32b",
    # "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5:1.5b-instruct",
    # "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen-2.5-coder-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "unsloth/DeepSeek-R1-Distill-Qwen-7B"
}


def load_model(model_name: str):
    global vllm_model, tokenizer
    if vllm_model is not None:
        return vllm_model, tokenizer
    # mapped_model = vllm_model_map.get(model_name)
    mapped_model = model_name
    tokenizer = AutoTokenizer.from_pretrained(mapped_model)
    vllm_model = LLM(mapped_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return vllm_model, tokenizer


def forward(
        inputs,
        model,
        tokenizer,
        max_new_tokens: int = 2048,
        temperature: float = 0.05,
        batch_size: int = 32,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False
):
    logging.debug("Generating model response")
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=num_of_results,
        seed=seed if seed is not None else None,
        top_p=1,
        stop=["</s>"]
    )
    outputs = model.generate(
        inputs,
        sampling_params=sampling_params,
    )
    all_results = []
    for idx, output in enumerate(outputs):
        result = []
        for seq_idx, multi_result_output in enumerate(output.outputs):
            logging.debug(f"Prompt {idx + 1} - Sequence {seq_idx + 1}: {multi_result_output.text.replace(tokenizer.pad_token, '')}")
            logging.debug("_" * 70)
            result.append(extract_cot_and_answer(multi_result_output.text, is_reasoning_model))
        all_results.append(result)
    logging.debug(f"Total number of results for the batch: {len(all_results)}")
    return all_results


def review_comment_generation(
        instance_index,
        model_name: str,
        test_name: str,
        shard_indices: list,
        base_dir: str,
        batch_size: int = 32,
        temperature: float = 0.7,
        pause_duration: int = 4,
        batch_call: bool = False,
        num_of_results: int = 1,
        seed: int = None,
        is_reasoning_model: bool = False
):
    shard_index = shard_indices[instance_index]
    print(f"Processing instance {instance_index} for shard {shard_index}")
    logging.info(f"Processing instance {instance_index} for shard {shard_index}")

    filtered_input, existing_results, output_path = get_unprocessed_examples(
        base_dir, model_name, test_name,
        shard_index, num_of_results,
        is_reasoning_model
    )

    model, tokenizer = load_model(model_name)
    batch_size = 2
    for i in tqdm(range(0, len(filtered_input), batch_size)):
        end_index = min(i + batch_size, len(filtered_input))
        batch = filtered_input[i:end_index]

        prompts = [v["prompt"] for v in batch]
        texts = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)

        print(f"Processing batch {i} to {end_index}")
        logging.debug(f"Processing batch {i} to {end_index}")
        try:
            results = forward(
                texts,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                batch_size=batch_size,
                num_of_results=num_of_results,
                seed=seed,
                is_reasoning_model=is_reasoning_model
            )
            save_results(batch, results, existing_results, output_path)
        except Exception as e:
            logging.error("Error: ", e)
            sleep(pause_duration)

    logging.info(f"Completed processing shard {shard_index}")
