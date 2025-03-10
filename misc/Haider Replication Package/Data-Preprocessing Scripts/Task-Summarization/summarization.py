from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import json
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
L = 508

def split_large_text(large_text, max_tokens, tokenizer):
    tokens = tokenizer.tokenize(large_text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='test.json', required=True, help='.jsonl')
    parser.add_argument('--output_path', type=str, default=None, required=False, help='output path')
    parser.add_argument('--testcase', type=int, default=1000000, required=False, help='number of testcases')
    args = parser.parse_args()
    file_path = args.file_path
    file = open(file_path, 'r')
    def read_one_test():
        code = file.readline()
        return code
        
    def generateSummary(code):
        # print(code)
        # print(len(code))
        if len(code) == 0:
            return ("No Summary Found", 0)
        chunk_size = L
        chunks = split_large_text(code, chunk_size, tokenizer)
        ret = ""
        cnt = 0
        # assert len(chunks) == 1
        for i in range(len(chunks)):
            # print(len(chunks[i]))
            try:
                input_ids = tokenizer(chunks[i], return_tensors="pt").input_ids
                # print(len(input_ids[0]))
                output = model.generate(input_ids, max_length=20)
                summary = tokenizer.decode(output[0], skip_special_tokens=True)
                if ret != "":
                    ret += " "
                ret += summary.replace('\n', ' ')
                cnt += 1
            except:
                continue
        return (ret, cnt)

    def removeEscapeCharacter(code):
        return bytes(code, "utf-8").decode("unicode_escape")

    if args.output_path is None:
        res = sys.stdout
    else:
        res = open(args.output_path, 'w')
    testCaseCount = 0
    while testCaseCount < args.testcase:
        jsonObject = read_one_test()
        if jsonObject:
            testCaseCount += 1
            jsonOb = json.loads(jsonObject)
            code = jsonOb['pre-summary']
            # code = removeEscapeCharacter(code)
            summary, chunks = generateSummary(code)
            res.write(json.dumps({"token_chunks": chunks, "summary": summary}))
            res.write("\n")
            print("Testcase: ", testCaseCount, end='\r')
            # res.write("\n\n\n")
            # print(summary)
        else:
            break
        
    print("Summarization complete")
    res.close()
    file.close()
