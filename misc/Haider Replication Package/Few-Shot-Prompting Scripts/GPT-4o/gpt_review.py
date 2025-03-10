import argparse
import os
import json
from rank_bm25 import BM25Okapi
from time import sleep
import numpy as np
import random
import openai

def makestr(lst):
    return lst
 
def dump(s):
    return s.replace("\n", " ")

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

def modify_cg(s):
    return s.replace(",", " , ").replace("->", " -> ")

def process_code(code):
    code = code.split("\n")[1:] 
    code = [line for line in code if len(line.strip()) > 0] 
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

def get_output(i, s):
    return dump(s) + '\n'

from evaluator.smooth_bleu import bleu_fromstr
def get_bleu(preds, golds):
    return bleu_fromstr([preds], [golds], rmstop=False)

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--open_key", default=None, type=str, required=True,
                        help="Enter API key")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="davinci/cushman/instruct") # instruct model option added
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

    print("In Debug mode: ", debug)
    # openai.api_key = args.open_key
    openai.base_url = "http://localhost:4000"


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
    if args.start_index==0:
        open(outdir+'/'+"preds.txt","w", encoding="utf-8").close()
        open(outdir+'/'+"preds_multiple.txt","w", encoding="utf-8").close()
        open(outdir+'/'+"golds.txt","w",encoding="utf-8").close()   
        open(outdir+'/'+"bleus.txt","w",encoding="utf-8").close()   
    

    ## Setting target model
    if args.model=="instruct_4_turbo_1":
        target_model="gpt-4-1106-preview"
    
    elif args.model=="instruct_4_turbo_2":
        target_model="gpt-4-0125-preview"

    elif args.model=="gpt_4o":
        target_model="gpt-4o"
    
    

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
    print(f"Model: {target_model}")
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
    error_count=0
    blank_count=0
    total_error=0
    while i-args.start_index<min(len(test_code),args.testcase):
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
    
            else:
                random.shuffle(indices)
                x=indices[0:shot]

            ### GPT-4 Turbo Model ####
            ##############################################################
            msg = []
            ##############################################################
            instruction = "Please Give FORMAL Codereview for software developers in one sentence for testcase, implementing Few Shot Learning from example. Dont start with Codereview/review. Just give the answer."
            msg.append({"role": "user", "content": instruction})
            
            ##############################################################
            for w in x:
                context=""
                context=context+"Code: \t"+train_code[w]+"\n"
                if args.with_summary==1:
                    context=context+"Summary: \t"+train_summary[w]+"\n"
                if args.with_callgraph==1:
                    context=context+"Callgraph: \t"+train_callgraph[w]+"\n"
                context=context+"Codereview: "
                msg.append({"role": "user", "content": context})
                context=train_nl[w] + " </s>"+"\n\n"
                msg.append({"role": "assistant", "content": context})
                

            ##############################################################
            
            context=""
            context=context+"Code: \t"+test_code[i]+"\n"
            if args.with_summary==1:
                context=context+"Summary: \t"+test_summary[i]+"\n"
            if args.with_callgraph==1:
                context=context+"Callgraph: \t"+test_callgraph[i]+"\n"

            context=context+"Codereview: "
            
            msg.append({"role": "user", "content": context})
           

            print("################context ####################")
            print(msg)
            fmul = open(outdir+'/'+"preds_multiple.txt","a", encoding="utf-8")
     
                
            if not debug:
                response = openai.chat.completions.create(
                  model=target_model,
                  messages = msg,
                  max_tokens=250,
                  temperature=.7,
                  n=5,
                  stop=['\n',"</s>"],
                  top_p=1, 
                  frequency_penalty=0,
                  presence_penalty=0,
                  seed=args.seed
                )

            ################## GPT-4 Turbo Model or GPT4o ############################


                print("Fingerprints: ", response.system_fingerprint)
                max_score=0
                modelout=""
                bleu_score=-1
                final_response = ""
                for choice in response.choices:
                    print("################model response ####################")
                    print(choice.message.content)
                    fmul.write(get_output(i, choice.message.content))
                    final_response = choice.message.content
                    # calculate the BLEU score
                    bleu_score = bleu_fromstr([final_response], [gold], rmstop=False)
                    print("BLEU score: ", bleu_score)
                    if bleu_score > max_score:
                        max_score = bleu_score
                        modelout =  final_response
            else:
                max_score = 0
                modelout = test_summary[i]


            fr = open(outdir+'/'+"preds.txt","a", encoding="utf-8")
            fg = open(outdir+'/'+"golds.txt","a", encoding="utf-8") 
            fb = open(outdir+'/'+"bleus.txt","a", encoding="utf-8")

            fr.write(get_output(i, modelout))
            fg.write(get_output(i, gold))
            fb.write(get_output(i, str(max_score)))
            fr.close()
            fg.close()
            fb.close()
            fmul.close()

            

            sleep(int(args.pause_duration))
            
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
            sleep(1)
            continue
        
    print("\n-----------------\n")
    print(f"Total blank responses: {blank_count}")
    print("\n-----------------\n")
    print(f"Total error responses: {total_error}")
    print("\n-----------------\n")
    

    
main()
