import argparse
import os
import json
from rank_bm25 import BM25Okapi
from time import sleep
import numpy as np
import random
import openai

def makestr(lst):
    #p=""
    #for w in lst:
        #p=p+w+" "
    return lst#p.strip()   
 
def dump(s):
    return s.replace("\n", " ")
    # return json.dumps(s)


def removeplusminus(s):
    ret = ''
    for i in s.split('\n'):
        if i == '':
            continue
        if i[0] == '-':
            ret += "[DEL]"
        elif i[0] == '+':
            ret += "[ADD]"
        else:
            ret += "[KEEP]"

        ret += i[1:] + '\n'
    return ret
    # ret = ''
    # for i in s.split('\n'):
    #     if i == '':
    #         continue
    #     ret += i[1:] + '\n'
    # return ret
def modify_cg(s):
    return s.replace(",", " , ").replace("->", " -> ")
    ret = ''
    for x in s.split("->"):
        # print(x)
        # input()
        ret += x + ' '
    return ret[:-1]

def process_code(code):
    # mimics the SimpleGenDataset -> __init__ and/or convert_examples_to_features function
    code = code.split("\n")[1:] # remove start @@
    code = [line for line in code if len(line.strip()) > 0] # remove empty lines
    map_dic = {"-": 0, "+": 1, " ": 2}
    def f(s):
        if s in map_dic:
            return map_dic[s]
        else:
            return 2
    labels = [f(line[0]) for line in code]
    code = [line[1:].strip() for line in code]
    inputstr = ""
    for label, line in zip(labels, code):
        if label == 1:
            inputstr += "<add>" + line
        elif label == 0:
            inputstr += "<delete>" + line
        else:
            inputstr += "<keep>" + line

    return inputstr

def modify(s):
    return process_code(s)
    # return s
    temp = s.split("@@")
    ret = ""
    for i in range(2, len(temp)):
        ret += temp[i] + "@@"
    if len(ret) > 0:
        ret = ret[:-2]
    return removeplusminus(ret)

def get_output(i, s):
    return dump(s) + '\n'
    # return str(i) + '\t' + dump(s) + '\n'


from evaluator.smooth_bleu import bleu_fromstr
def get_bleu(preds, golds):
    # chars = "(_)`."
    # for c in chars:
    #     preds = preds.replace(c, " " + c + " ")
    #     preds = " ".join(preds.split())
    #     golds = golds.replace(c, " " + c + " ")
    #     golds = " ".join(golds.split())
    return bleu_fromstr([preds], [golds], rmstop=False)

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--open_key", default=None, type=str, required=True,
                        help="Enter API key")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="davinci/cushman/instruct") # instruct model option added
    #parser.add_argument("--data_folder", default=None, type=str, required=True,
                        #help="data folder path ")
    parser.add_argument("--pause_duration", default=None, type=str, required=True,
                        help="time to stop between samples") 
    parser.add_argument("--mode", default=None, type=str, required=True,
                        help="random/BM25")  
    parser.add_argument("--number_of_fewshot_sample", default=None, type=str, required=True,
                        help="1,2,4,6,8") 
    parser.add_argument("--language", default='cpp', type=str, required=False,
                        help="csharp/cpp")
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="train file path")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="test file path")
    parser.add_argument("--testcase", default=1, type=int, required=False,
                        help="testcase number")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="output directory")
    parser.add_argument("--debug", default=1, type=int, required=False,
                        help="debug mode")
    parser.add_argument("--with_callgraph", default=1, type=int, required=False,
                        help="with callgraph")
    parser.add_argument("--with_summary", default=1, type=int, required=False,
                        help="with summary")
    parser.add_argument("--number_of_results", default=1, type=int, required=False,
                        help="number of results")
    parser.add_argument("--start_index", default=0, type=int, required=False,
                        help="start index")
    parser.add_argument("--seed", default=42, type=int, required=False, help="seed")
    
    args = parser.parse_args()
    debug = args.debug
    number_of_results = args.number_of_results
    
    print("In Debug mode: ", debug)
    openai.api_key = args.open_key
    
    ## Setting seed
    # random.seed(args.seed)

    ## Setting output directory
    last = ""
    if debug:
        last = "_debug"+last
    if args.with_callgraph==1:
        last = "_cg"+last
    if args.with_summary==1:
        last = "_sum"+last
    outdir = args.output_dir + "/" + args.model+"_"+args.mode+"_"+args.number_of_fewshot_sample+last
    #making necessary forlders
    if not os.path.exists(outdir):
       os.makedirs(outdir)
    open(outdir+'/'+"preds.txt","w", encoding="utf-8").close()
    open(outdir+'/'+"golds.txt","w",encoding="utf-8").close()   
    open(outdir+'/'+"bleus.txt","w",encoding="utf-8").close()   
    

    ## Setting target model
    if args.model=="davinci":
        target_model="code-davinci-002"
    elif args.model=="cushman":
        target_model="code-cushman-001"
    elif args.model=="instruct": # turbo instruct model option added
        target_model="gpt-3.5-turbo-instruct"
    
    # target_model = "gpt-3.5-turbo-16k-0613"

    ##Reading data
    train_json = []
    for line in open(args.train_file, 'r', encoding="utf-8"):
        train_json.append(json.loads(line))
    print("\n-----------------\n")
    print(f"Train data length: {len(train_json)}")
    print("\n-----------------\n")
    
    test_json = []
    for line in open(args.test_file, 'r', encoding="utf-8"):
        test_json.append(json.loads(line))
    print("\n-----------------\n")
    print(f"Test data length: {len(test_json)}")
    print("\n-----------------\n")

    
    ##Processing data
    train_code = []
    train_nl = []
    train_callgraph = []
    train_summary = []
    for i in range(len(train_json)):
        train_nl.append(makestr(train_json[i]['msg']))    
        train_summary.append(makestr(train_json[i]['summary']))
        train_callgraph.append(modify_cg(makestr(train_json[i]['callgraph'])))
        train_code.append(modify(makestr(train_json[i]['patch'])))

    
    test_code = []
    test_nl = []
    test_callgraph = []
    test_summary = []
    for i in range(len(test_json)):
        test_nl.append(makestr(test_json[i]['msg']))    
        test_summary.append(makestr(test_json[i]['summary']))
        test_callgraph.append(modify_cg(makestr(test_json[i]['callgraph'])))
        test_code.append(modify(makestr(test_json[i]['patch'])))

    print("\n-----------------\n")
    print(f"Mode: {args.mode}")
    print(f"Number of fewshot samples: {args.number_of_fewshot_sample}")
    print(f"Model: {args.model}")
    print(f"Pause duration: {args.pause_duration}")
    print(f"Testcase: {args.testcase}")
    print("\n-----------------\n")

    ##Setting BM25
    if args.mode=="BM25":
        tokenized_corpus = [doc.split(" ") for doc in train_code]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        indices=[]
        for i in range(len(train_code)):
            indices.append(i)
     
    i=args.start_index
    is_error=0
    error_count=0
    blank_count=0
    total_error=0
    while i-args.start_index<args.testcase and i<len(test_code):
        print(f"Testcase: {i} running", end="\r")
        shot=int(args.number_of_fewshot_sample) 
        if shot<1:
            shot=1
        try:
            query=test_code[i]
            gold=test_nl[i]
            tokenized_query = query.split(" ")
            if args.mode=="BM25":
                x=bm25.get_scores(tokenized_query)   
                arr = np.array(x)
                x=arr.argsort()[-shot:][::-1]
                # print("inBM25")
            else:
                random.shuffle(indices)
                x=indices[0:shot]
            
            # print(f"Fewshot samples: {x}")
            
            
            # context="Provide a detailed response to:\n\n\n"
            # context="Generate a one-sentence response to:\n\n"
            context=""
            for w in x:
                context=context+"[CODE]\t"+train_code[w]+"\n"
                if args.with_summary==1:
                    context=context+"[SUMMARY]\t"+train_summary[w]+"\n"
                if args.with_callgraph==1:
                    context=context+"[CALLGRAPH]\t"+train_callgraph[w]+"\n"
                context=context+"[CODEREVIEW]\t<s>"+ train_nl[w] + " </s>"+"\n\n"
            
            
            #print(context)
            #print(test_code[i])
            context=context+"[CODE]\t"+test_code[i]+"\n"
            if args.with_summary==1:
                context=context+"[SUMMARY]\t"+test_summary[i]+"\n"
            if args.with_callgraph==1:
                context=context+"[CALLGRAPH]\t"+test_callgraph[i]+"\n"
            context=context+"[CODEREVIEW]\t<s>"
            if debug:
                print(len(context))
                print(context)
            # print("About to call api ")
            if not debug:
                response = openai.Completion.create(
                engine=target_model,
                prompt=context,
                temperature=0.7,
                max_tokens=250,
                stop=['\n',"</s>"],
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=number_of_results,
                seed=args.seed
                )
                # print("#####################model response###############")
                # print(f"Number of Responses: {len(response.choices)}")
                # print("##################################################")
                modelout=""
                bleu_score=-1
                for j in range(len(response.choices)):
                    # print(f"Response {j}: {response.choices[j].text}")
                    resp = response.choices[j].text.split("</s>")[0]
                    new_bleu_score = get_bleu(resp, gold)
                    # print(f"BLEU: {new_bleu_score}")
                    # print("--------------------------------------------------")
                    if new_bleu_score > bleu_score:
                        bleu_score = new_bleu_score
                        modelout = resp
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # print(f"Final Model output: {modelout}")
                # print(f"Final BLEU score: {bleu_score}")
            else:
                # modelout = "test"
                bleu_score = 0
                modelout = test_summary[i]

            fr = open(outdir+'/'+"preds.txt","a", encoding="utf-8")
            fg = open(outdir+'/'+"golds.txt","a", encoding="utf-8") 
            fb = open(outdir+'/'+"bleus.txt","a", encoding="utf-8")

            fr.write(get_output(i, modelout))
            fg.write(get_output(i, gold))
            fb.write(get_output(i, str(bleu_score)))
            fr.close()
            fg.close()
            fb.close()
            
            # print("going_sleep")
            sleep(int(args.pause_duration))
            # print("wakeup")
            
            if modelout=="":
                blank_count=blank_count+1
            
            i=i+1
            error_count=0
            is_error=0
            
        except Exception as e:
            print("Error: ",e)
            is_error=1
            error_count=error_count+1
            total_error=total_error+1
            print(error_count)
            if error_count==5: # change this value to 5 from 10 
                fr = open(outdir+'/'+"preds.txt","a", encoding="utf-8")
                fg = open(outdir+'/'+"golds.txt","a",encoding="utf-8")   
                fb = open(outdir+'/'+"bleus.txt","a",encoding="utf-8")
                fr.write(get_output(i, ""))
                fg.write(get_output(i, gold))
                fb.write(get_output(i, "0.0"))
                fr.close()
                fg.close()
                fb.close()
                i=i+1
                blank_count=blank_count+1
                error_count=0
                is_error=0
            sleep(5)
            continue
        
    print("\n-----------------\n")
    print(f"Total blank responses: {blank_count}")
    print("\n-----------------\n")
    print(f"Total error responses: {total_error}")
    print("\n-----------------\n")
    

    
main()
