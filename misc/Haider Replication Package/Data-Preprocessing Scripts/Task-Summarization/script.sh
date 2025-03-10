#!/bin/bash
python3 ./summarization.py --file_path msg-test-5000-presummary.jsonl \
			   --output msg-test-5000-summary.jsonl
			   
python3 ./summarization.py --file_path msg-test-presummary.jsonl \
			   --output msg-test-summary.jsonl
			   
python3 ./summarization.py --file_path msg-valid-presummary.jsonl \
			   --output msg-valid-summary.jsonl
			   
python3 ./summarization.py --file_path msg-train-presummary.jsonl \
			   --output msg-train-summary.jsonl
