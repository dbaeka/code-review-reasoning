model="gemini"
pause_duration=30
mode="BM25"
number_of_fewshot_sample=5
number_of_results=5
testcase=5000
with_summary=1
with_callgraph=1

data_dir="../../../Organized/datasets/task2/merged/"
output_dir="../../../Organized/datasets/task2/output"
train_file="${data_dir}msg-train-merged.jsonl"
test_file="${data_dir}msg-test-5000-merged.jsonl"
########################################################
API_KEY="YOUR_API_KEY"
seed=0
debug=0
start_index=0

path="${output_dir}/${model}_${mode}_${number_of_fewshot_sample}"

if (( with_summary == 1 )); then
    path="${path}_sum"
fi
if (( with_callgraph == 1 )); then
    path="${path}_cg"
fi


python3 ./process.py --open_key $API_KEY \
                       --model $model \
                       --pause_duration $pause_duration \
                       --mode $mode \
                       --number_of_fewshot_sample $number_of_fewshot_sample \
                       --train_file $train_file \
                       --test_file  $test_file \
                       --testcase $testcase \
                       --output_dir $output_dir \
                       --debug $debug \
                       --with_summary $with_summary \
                       --with_callgraph $with_callgraph \
                       --number_of_results $number_of_results \
                       --start_index $start_index \
                       --seed $seed 2>&1| tee $path/reports.log

