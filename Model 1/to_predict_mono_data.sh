#!/bin/bash

echo "################ Starting ################"
set -e

FAIRSEQ_DIR=fairseq

######## Command line arguments ########
TGT_FILE=$1
SRC_FILE=$2

DATA_DIR=$3
SAVE_DIR=$4

EPOCHS=$5

MODE=$6

NORM_PUNC=mosesdecoder/scripts/tokenizer/normalize-punctuation.perl
TOKENIZER=mosesdecoder/scripts/tokenizer/tokenizer.perl
DETOKENIZER=mosesdecoder/scripts/tokenizer/detokenizer.perl
FAIRSEQ_PREPROCESS=fairseq-preprocess
FAIRSEQ_TRAIN=fairseq-train
FAIRSEQ_GENERATE=fairseq-generate

EVALUATE="python evaluate.py"

MODELS=${SAVE_DIR}/models/${TGT_FILE}_${SRC_FILE}
CHECKPOINTS=${SAVE_DIR}/checkpoints/${TGT_FILE}_${SRC_FILE}
LOGS=${SAVE_DIR}/logs/${TGT_FILE}_${SRC_FILE}
TRANSLATIONS=${SAVE_DIR}/translations/${TGT_FILE}_${SRC_FILE}
DATA_OUT=${SAVE_DIR}/data_out/${TGT_FILE}_${SRC_FILE}

#mkdir -p ${MODELS} ${CHECKPOINTS} ${LOGS} ${DATA_OUT} ${TRANSLATIONS}
#
#
#
#echo "################ Tokenizing and preprocessing data ################"
#
#TRUECASER=mosesdecoder/scripts/recaser/truecase.perl
#
#
#for file in ${DATA_DIR}/*.${SRC_FILE}; do
#  echo $file
#  cat $file \
#  | $NORM_PUNC -l $SRC_FILE \
#  > ${DATA_DIR}/$(basename $file).normpunc
#  cp ${DATA_DIR}/$(basename $file).normpunc ${DATA_DIR}/$(basename $file)
#  rm ${DATA_DIR}/$(basename $file).normpunc
#done
#
#for file in ${DATA_DIR}/*.${TGT_FILE}; do
#  echo $file
#  cat $file \
#  | $NORM_PUNC -l $TGT_FILE \
#  > ${DATA_DIR}/$(basename $file).normpunc
#  cp ${DATA_DIR}/$(basename $file).normpunc ${DATA_DIR}/$(basename $file)
#  rm ${DATA_DIR}/$(basename $file).normpunc
#done
#
#echo "NORMALIZED PUNC"
#
#
#
#for file in ${DATA_DIR}/*.${SRC_FILE}; do
#
#  $TRUECASER --model truecase-model.${SRC_FILE} < ${DATA_DIR}/$(basename $file) > ${DATA_DIR}/$(basename $file).tok
#
#  wc -l ${DATA_DIR}/$(basename $file).tok
#
#  if [[ "$file" = *"mt"* ]]; then
#    python tokenisemt.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
#  elif [[ "$file" = *"ic"* ]]; then
#    python tokeniseic.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
#  else
#    cat ${DATA_DIR}/$(basename $file).tok \
#    | $TOKENIZER -l $SRC_FILE \
#    > ${DATA_DIR}/$(basename $file)
#  fi
#
#  wc -l ${DATA_DIR}/$(basename $file)
#  rm ${DATA_DIR}/$(basename $file).tok
#  echo ${DATA_DIR}/$(basename $file)
#done
#
#for file in ${DATA_DIR}/*.${TGT_FILE}; do
#  wc -l ${DATA_DIR}/$(basename $file)
#  $TRUECASER --model truecase-model.${TGT_FILE} < ${DATA_DIR}/$(basename $file) > ${DATA_DIR}/$(basename $file).tok
#
#  wc -l ${DATA_DIR}/$(basename $file).tok
#
#  if [[ "$file" = *"mt"* ]]; then
#    python tokenisemt.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
#  elif [[ "$file" = *"ic"* ]]; then
#    python tokeniseic.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
#  else
#    cat ${DATA_DIR}/$(basename $file).tok \
#    | $TOKENIZER -l $SRC_FILE \
#    > ${DATA_DIR}/$(basename $file)
#  fi
#
#  rm ${DATA_DIR}/$(basename $file).tok
#  wc -l ${DATA_DIR}/$(basename $file)
#  echo ${DATA_DIR}/$(basename $file)
#done
#
#echo "applied truecasing and tokenized train/dev/test"
#
#echo "applying BPE next:"
#python sp_tools.py \
#  --encode \
#  --src ${SRC_FILE} \
#  --tgt ${TGT_FILE} \
#  --data_dir ${DATA_DIR}/ \
#  --data_out ${DATA_OUT}/ \
#  --model_dir /netscratch/abela/dfkibaseline/results/models/en_${SRC_FILE}/ \
#  --vocab_size 5000
#
#echo "################ Done tokenizing ################"
#
#
#
#
#
#
#
#
#echo translating $DATA_OUT/test.bpe.${SRC_FILE} from $SRC_FILE to $TGT_FILE and storing to  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp using /netscratch/abela/dfkibaseline/results/checkpoints/en_${SRC_FILE}/checkpoint_best.pt and $DATA_OUT
#
#cat $DATA_DIR/test.${SRC_FILE} \
#|   fairseq-interactive  /netscratch/abela/dfkibaseline/results/data_out/en_${SRC_FILE}/ --input=$DATA_OUT/test.bpe.${SRC_FILE} --path /netscratch/abela/dfkibaseline/results/checkpoints/en_${SRC_FILE}/checkpoint_best.pt -s $SRC_FILE -t $TGT_FILE --beam 3 --remove-bpe --max-tokens 3000 --buffer-size 3000 --max-source-positions 128 --max-target-positions 128 --skip-invalid-size-inputs-valid-test \
#> $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out
#
#
#
#echo "################ Done fairseq-generate ################"
#
#cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp
#
#
#cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out | grep -P "^S" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp.src
#
#python sp_tools.py \
#  --decode \
#  --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp.src \
#  --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
#  --model_dir ${MODELS}/
#
#python sp_tools.py \
#  --decode \
#  --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp \
#  --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe2.hyp \
#  --model_dir ${MODELS}/
#
#
#
#
#if [[ ${SRC_FILE} = "en" ]]; then
#    cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
#    | $DETOKENIZER -l $SRC_FILE \
#    > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src
#    
#    
#    cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe2.hyp \
#    | $DETOKENIZER -l $SRC_FILE \
#    > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.tgt
#
#    echo "Done"
#
#elif [[ ${SRC_FILE} = "ic" ]]; then
#    python tokeniseic.py \
#        --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
#        --tgt  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src \
#        --decode
#    python tokeniseic.py \
#        --src $DATA_DIR/test.${SRC_FILE} \
#        --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.originalSrc \
#        --decode
#else
#    python tokenisemt.py \
#        --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
#        --tgt  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src \
#        --decode
#    python tokenisemt.py \
#        --src $DATA_DIR/test.${SRC_FILE} \
#        --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.originalSrc \
#        --decode
#fi


cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe2.hyp\
| $DETOKENIZER -l $SRC_FILE \
> $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.hyp





echo "################ Done  ################"


