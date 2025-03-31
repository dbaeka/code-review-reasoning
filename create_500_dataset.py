from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
    for ds in ["_base", "_summary", "_callgraph", "_summary_callgraph"]:
        dataset = load_dataset(f"dbaeka/soen_691_few_shot_test_5000{ds}_hashed")['test']
        dataset = dataset.select(range(500))
        dataset.push_to_hub(f"dbaeka/soen_691_few_shot_test_500{ds}_hashed")
