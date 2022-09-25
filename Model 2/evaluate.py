import sacrebleu
import argparse
from subprocess import call
import os
from pathlib import Path

SRC_LANG = "mt"
TGT_LANG = "en"


# +
pathlist = Path("test_dataset/data/").glob('**/predictions*.' + TGT_LANG)
with open("test_dataset/data/actual_test." + TGT_LANG, "w+", encoding="utf-8") as actual_test:
    with open("test_dataset/data/actual_preds." + TGT_LANG, "w+", encoding="utf-8")  as actual_preds:
        for path in pathlist:
            print(path)
            line_count = 0
            pred_file_num = str(path)[-5:-3]
            try:
                with open("test_dataset/data/predictions" + pred_file_num + "." + TGT_LANG, 'r+', encoding="utf-8") as pred_file:
                    with open("test_dataset/data/actualTest" + pred_file_num + "." + TGT_LANG, "r+", encoding="utf-8") as test_file:
                        test_file_lines = []
                        for line in test_file:
                            test_file_lines.append(line)

                        for i, line in enumerate(pred_file):

                            if test_file_lines[i] == "\n":
                                print("HERE")
                            if line != "\n" and test_file_lines[i] != "\n":
                                actual_preds.write(line.replace("\n", ""))
                                actual_preds.write("\n")
    #                             print(line.replace("\n", ""))
                                actual_test.write(test_file_lines[i].replace("\n", ""))   
                                actual_test.write("\n") 
            except:
                continue
                        
                        
with open("test_dataset/data/actual_test." + TGT_LANG, "r+", encoding="utf-8") as actual_test:
    with open("test_dataset/data/actual_preds." + TGT_LANG, "r+", encoding="utf-8")  as actual_preds:
        line_count = 0
        for line in actual_test:                 
            if line != "\n":
                line_count += 1
        print(line_count)
        line_count = 0
        for line in actual_preds:                 
            if line != "\n":
                line_count += 1
        print(line_count)
# -

if TGT_LANG == "mt":
    os.system(" python tokenisemt.py --src test_dataset/data/actual_preds.mt --tgt test_dataset/data/actual_preds_detok.mt --decode")
    os.system(" python tokenisemt.py --src test_dataset/data/actual_test.mt --tgt test_dataset/data/actual_test_tok.mt --encode")
    os.system(" python tokenisemt.py --src test_dataset/data/actual_test_tok.mt --tgt test_dataset/data/actual_test.mt --decode")
elif TGT_LANG == "ic":
    os.system(" python tokeniseic.py --src test_dataset/data/actual_preds.ic --tgt test_dataset/data/actual_preds_detok.ic --decode")
elif TGT_LANG == "en":
    os.system("test_dataset/data/detokenizer.perl  < test_dataset/data/actual_preds.en  > test_dataset/data/actual_preds_detok.en")

system_lines = []
system_output="test_dataset/data/actual_preds_detok." + TGT_LANG
gold_reference="test_dataset/data/actual_test." + TGT_LANG

command = "wc -l " + gold_reference
res = call(command, shell=True)

command = "wc -l " + system_output
res = call(command, shell=True)

command = "sacrebleu " + gold_reference + " -i " + system_output + " -m bleu -w 4 --encoding utf-8"
res = call(command, shell=True)


command = "sacrebleu " + gold_reference + " -i " + system_output + " -m chrf -w 4 --encoding utf-8"
res = call(command, shell=True)



