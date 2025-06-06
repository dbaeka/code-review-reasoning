#!/bin/bash
dir="../Organized/datasets/task2/"
files=("msg-test"\ "msg-test-5000"\ "msg-valid"\ "msg-train")
append="-callgraph-inside-scope.jsonl"

for file in $files
do
    echo "Processing $file"
    
    python3 callGraph.py --file_path ${dir}original-set/${file}.jsonl \
			 --output ${dir}callgraph_within_scope_new/${file}${append} \
			 --debug 0
done 