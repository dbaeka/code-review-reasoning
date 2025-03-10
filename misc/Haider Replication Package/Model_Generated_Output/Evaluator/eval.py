
from bert_score import score
import argparse
from evaluator.smooth_bleu import bleu_fromstr
import torch
def get_bleu(preds, golds):
    # chars = "(_)`."
    # for c in chars:
    #     preds = preds.replace(c, " " + c + " ")
    #     preds = " ".join(preds.split())
    #     golds = golds.replace(c, " " + c + " ")
    #     golds = " ".join(golds.split())
    return bleu_fromstr([preds], [golds], rmstop=False)

def em(preds, golds):
    em = 0
    for pred, gold in zip(preds, golds):
        if " ".join(pred.split()) == " ".join(gold.split()):
            em += 1
    em = em / len(golds)
    return em


parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, type=str, required=True,
                    help="davinci/cushman/instruct") # instruct model option added
parser.add_argument("--mode", default=None, type=str, required=True,
                    help="random/BM25")  
parser.add_argument("--number_of_fewshot_sample", default=None, type=str, required=True,
                    help="1,2,4,6,8") 
parser.add_argument("--output_dir", default=None, type=str, required=False,
                    help="output directory")
parser.add_argument("--with_callgraph", default=1, type=int, required=False,
                    help="with callgraph")
parser.add_argument("--with_summary", default=1, type=int, required=False,
                    help="with summary")
parser.add_argument("--testcase", default=-1, type=int, required=False)
args = parser.parse_args()
last = ""
if args.with_callgraph==1:
    last = "_cg"+last
if args.with_summary==1:
    last = "_sum"+last
outdir = args.output_dir + "/" + args.model+"_"+args.mode+"_"+args.number_of_fewshot_sample+last
#making necessary forlders
preds = []
with open(outdir+'/'+"preds.txt","r", encoding="utf-8") as f:
    for line in f:
        preds.append(line.strip())
golds = []
with open(outdir+'/'+"golds.txt","r", encoding="utf-8") as f:
    for line in f:
        golds.append(line.strip())
print(len(preds), len(golds))
assert len(preds) == len(golds)
if args.testcase != -1:
    preds = preds[:args.testcase]
    golds = golds[:args.testcase]
print(len(preds), len(golds))
blank_count = 0
for pred in preds:
    if pred.strip() == "":
        blank_count += 1
bert = 0
chunk = 100
for i in range(0, len(preds), chunk):
    P, R, F1 = score(cands=preds[i : min(len(preds), i + chunk)], refs=golds[i : min(len(preds), i + chunk)], lang="en")
    bert += F1.sum()
    torch.cuda.empty_cache()
print("\n-----------------\n")
print(f"The number of samples: {len(preds)}")
print("\n-----------------\n")
print(f"Total blank responses: {blank_count}")
print("\n-----------------\n")
print(f"BERTScore: {bert/len(preds):.4f}")
print("\n-----------------\n")
print(f"Bleu score: {bleu_fromstr(preds, golds, rmstop=False)}")
print("\n-----------------\n")
print(f"Bleu score (with rmStop): {bleu_fromstr(preds, golds, rmstop=True)}")
print("\n-----------------\n")
print(f"EM: {em(preds, golds)}")
print("\n-----------------\n")

print("Ignoring Blank Lines\n")
print(f"BERTScore: {bert/(len(preds) - blank_count):.4f}")
print("\n-----------------\n")
print(f"Bleu score: {(bleu_fromstr(preds, golds, rmstop=False) * len(preds))/(len(preds) - blank_count)}")
print("\n-----------------\n")
print(f"EM: {(em(preds, golds) * len(preds))/(len(preds) - blank_count)}")
print("\n-----------------\n")