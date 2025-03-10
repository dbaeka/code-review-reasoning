model="instruct"
mode="BM25"
number_of_fewshot_sample=5
with_summary=1
with_callgraph=1
testcase=5000
output_dir="../../../Organized/datasets/task2/output"
########################################################
python3 ./eval.py      --model $model \
                       --mode $mode \
                       --number_of_fewshot_sample $number_of_fewshot_sample \
                       --output_dir $output_dir \
                       --with_summary $with_summary \
                       --with_callgraph $with_callgraph \
                       --testcase $testcase