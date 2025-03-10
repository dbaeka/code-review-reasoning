from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import json
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')

    # file_path = sys.argv[1]
    file_path = "test2.txt"
    res = open('result2.txt', 'w')
    file = open(file_path, 'r')
    def read_one_test():
        # code = file.readline()
        code = file.read()
        return code
        
    def generateSummary(code):
        chunk_size = 512
        chunks = [code[i:min(i + chunk_size, len(code))] for i in range(0, len(code), chunk_size)]

        ret = ""
        for chunk in chunks:
            input_ids = tokenizer(chunk, return_tensors="pt").input_ids
            output = model.generate(input_ids, max_length=20)
            summary = tokenizer.decode(output[0], skip_special_tokens=True)
            if(len(ret) > 0):
                ret += " "
            ret += summary.replace("\n", " ") + "\n"
        return ret

    def removeEscapeCharacter(code):
        return bytes(code, "utf-8").decode("unicode_escape")

    while True:
        line = read_one_test()
        if line:
            # code = removeEscapeCharacter(code)
            summary = generateSummary(line)
            res.write(summary)
            res.write("\n")
            # res.write("\n\n\n")
            # print(summary)
        else:
            break
        
    print("Summarization complete")
    res.close()
    file.close()