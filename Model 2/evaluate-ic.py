import sacrebleu
import argparse
from subprocess import call
import os
from pathlib import Path

SRC_LANG = "ic"
TGT_LANG = "en"


# +
pathlist = Path("/netscratch/abela/transformerbaseline/ic_en_dataset/data/").glob('**/predictions*.' + TGT_LANG)

with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/actual_test." + TGT_LANG, "w+", encoding="utf-8") as actual_test:
    with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/actual_preds." + TGT_LANG, "w+", encoding="utf-8")  as actual_preds:
        for path in pathlist:
            print(path)
            line_count = 0
            pred_file_num = str(path)[-5:-3]
#             try:
            with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/predictions" + pred_file_num+ "." + TGT_LANG, 'r', encoding="utf-8") as pred_file:
                with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/actualTest" + pred_file_num + "." + TGT_LANG, "r", encoding="utf-8") as test_file:
#                     print("HERE")
                    test_file_lines = []
                    for line in test_file:
                        test_file_lines.append(line)

                    for i, line in enumerate(pred_file):
#                         print(line)
                        if line != "\n" and test_file_lines[i] != "\n":
                            actual_preds.write(line.replace("\n", ""))
                            actual_preds.write("\n")
#                             print(line.replace("\n", ""))
                            actual_test.write(test_file_lines[i].replace("\n", ""))   
                            actual_test.write("\n")   
#                             print(test_file_lines[i])
#             except:
#                 continue
                              
#             actual_preds.write("\n")

                        
                        
with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/actual_test." + TGT_LANG, "r+", encoding="utf-8") as actual_test:
    with open("/netscratch/abela/transformerbaseline/ic_en_dataset/data/actual_preds." + TGT_LANG, "r+", encoding="utf-8")  as actual_preds:
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

if TGT_LANG == "ic":
    os.system(" python tokeniseic.py --src ic_en_dataset/data/actual_preds.ic --tgt ic_en_dataset/data/actual_preds_detok.ic --decode")
elif TGT_LANG == "en":
    os.system("test_dataset/data/detokenizer.perl -l en < ic_en_dataset/data/actual_preds.en  > ic_en_dataset/data/actual_preds_detok.en")

system_lines = []
system_output="ic_en_dataset/data/actual_preds_detok." + TGT_LANG
gold_reference="ic_en_dataset/data/actual_test." + TGT_LANG

command = "wc -l " + gold_reference
res = call(command, shell=True)

command = "wc -l " + system_output
res = call(command, shell=True)

command = "sacrebleu " + gold_reference + " -i " + system_output + " -m bleu chrf -w 4  --encoding utf-8"
res = call(command, shell=True)


command = "sacrebleu " + gold_reference + " -i " + system_output + " -m chrf -w 4 --encoding utf-8"
res = call(command, shell=True)



