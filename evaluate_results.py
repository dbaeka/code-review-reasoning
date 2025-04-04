import logging

import transformers
from datasets import load_dataset

from src.common.eval import bleu_fromstr, calculate_exact_match_batch, calculate_bert_score_batch

# hide the loading messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

if __name__ == "__main__":
    columns_to_skip = ['hash', 'few_shot_prompt', 'zero_shot_prompt', 'gold']

    test_names = ["test_500", "test_5000"]
    for test_name in test_names:
        pred_dataset = load_dataset(f"dbaeka/soen_691_{test_name}_final_selected_results")['test']

        print("PROCESSING ", test_name)

        for column in pred_dataset.column_names:
            if column in columns_to_skip:
                continue

            print(f"Evaluating {column}")
            golds = pred_dataset["gold"]
            preds = pred_dataset[column]

            pred_values = [pred["answer"] for pred in preds]

            bert_score = calculate_bert_score_batch(pred_values, golds)

            blank_count = 0
            for pred in pred_values:
                if pred.strip() == "":
                    blank_count += 1

            print("\n-----------------\n")
            print(f"The number of samples: {len(pred_values)}")
            print(f"Total blank responses: {blank_count}")
            print("\n-----------------\n")
            print(f"BERTScore: {bert_score / len(pred_values):.4f}")
            print(f"Bleu score: {bleu_fromstr(pred_values, golds, rmstop=False)}")
            print(f"Bleu score (with rmStop): {bleu_fromstr(pred_values, golds, rmstop=True)}")
            print(f"EM: {calculate_exact_match_batch(pred_values, golds)}")
            print("\n-----------------\n")

            print("Ignoring Blank Lines\n")
            print(f"BERTScore: {bert_score / (len(pred_values) - blank_count):.4f}")
            print(
                f"Bleu score: {(bleu_fromstr(pred_values, golds, rmstop=False) * len(pred_values)) / (len(pred_values) -
                                                                                                      blank_count)}")
            print(f"EM: {(calculate_exact_match_batch(pred_values, golds) * len(pred_values)) / (len(pred_values) -
                                                                                                 blank_count)}")
            print("\n-----------------\n")
