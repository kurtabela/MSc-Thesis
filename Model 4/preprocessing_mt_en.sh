#!/bin/bash

set -e

FAIRSEQ_DIR=fairseq

######## Command line arguments ########
TGT_FILE=$1
SRC_FILE=$2

DATA_DIR=$3/${SRC_FILE}_${TGT_FILE}
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

MODELS=${SAVE_DIR}/models/${SRC_FILE}_${TGT_FILE}
CHECKPOINTS=${SAVE_DIR}/checkpoints/${SRC_FILE}_${TGT_FILE}
LOGS=${SAVE_DIR}/logs/${SRC_FILE}_${TGT_FILE}
TRANSLATIONS=${SAVE_DIR}/translations/${SRC_FILE}_${TGT_FILE}
DATA_OUT=${SAVE_DIR}/data_out/${SRC_FILE}_${TGT_FILE}


cd secoco/fairseq 
pip install . 
cd ../../
pwd


pip list

if [[ ${SRC_FILE} == "mt" || ${TGT_FILE} == "mt" ]]; then

    #rm -rf results/
    rm -rf data/${SRC_FILE}_${TGT_FILE}/
    #
    mkdir -p data/${SRC_FILE}_${TGT_FILE}/
    cp -r original_data/${SRC_FILE}_${TGT_FILE}/* ${DATA_DIR}/
    mkdir -p ${MODELS} ${CHECKPOINTS} ${LOGS} ${DATA_OUT} ${TRANSLATIONS}

elif [[ ${SRC_FILE} == "ic" || ${TGT_FILE} == "ic" ]]; then
    #rm -rf results/
    rm -rf data/${SRC_FILE}_${TGT_FILE}/
    #
    mkdir -p data/${SRC_FILE}_${TGT_FILE}/
    cp -r original_data/${SRC_FILE}_${TGT_FILE}/* ${DATA_DIR}/
    mkdir -p ${MODELS} ${CHECKPOINTS} ${LOGS} ${DATA_OUT} ${TRANSLATIONS}

fi

for file in ${DATA_DIR}/*.${SRC_FILE}; do
  if [[ "$file" = *"dev"* ]]; then
     echo $file
     iconv -f utf16 -t utf8 -o "$file.new" "$file" &&
       mv -f "$file.new" "$file"
  fi
done

for file in ${DATA_DIR}/*.${TGT_FILE}; do
   if [[ "$file" = *"dev"* ]]; then
     echo $file
     iconv -f utf16 -t utf8 -o "$file.new" "$file" &&
       mv -f "$file.new" "$file"
   fi
done



echo "################ Training SentencePiece tokenizer ################"
echo ${SRC_FILE}
python sp_tools.py \
  --train \
  --src ${SRC_FILE} \
  --tgt ${TGT_FILE} \
  --data_dir ${DATA_DIR}/ \
  --model_dir ${MODELS}/ \
  --vocab_size 32000

echo "################ Done training ################"
#Vocab size may need to be changed depending on training files (3560 for pilot data train)

tail -n +4 ${MODELS}/sentencepiece.bpe.vocab | cut -f1 | sed 's/$/ 100/g' > $MODELS/fairseq.dict

echo "################ Tokenizing and preprocessing data ################"

TRAIN_TRUECASER=mosesdecoder/scripts/recaser/train-truecaser.perl
TRUECASER=mosesdecoder/scripts/recaser/truecase.perl


for file in ${DATA_DIR}/*.${SRC_FILE}; do
  echo $file
  cat $file \
  | $NORM_PUNC -l $SRC_FILE \
  > ${DATA_DIR}/$(basename $file).normpunc
  mv ${DATA_DIR}/$(basename $file).normpunc ${DATA_DIR}/$(basename $file)
done

for file in ${DATA_DIR}/*.${TGT_FILE}; do
  echo $file
  cat $file \
  | $NORM_PUNC -l $TGT_FILE \
  > ${DATA_DIR}/$(basename $file).normpunc
  mv ${DATA_DIR}/$(basename $file).normpunc ${DATA_DIR}/$(basename $file)
done

echo "NORMALIZED PUNC"


$TRAIN_TRUECASER --model truecase-model.${SRC_FILE} --corpus ${DATA_DIR}/train.${SRC_FILE}
$TRAIN_TRUECASER --model truecase-model.${TGT_FILE} --corpus ${DATA_DIR}/train.${TGT_FILE}

echo "TRAINED TRUECASER"


for file in ${DATA_DIR}/*.${SRC_FILE}; do

  $TRUECASER --model truecase-model.${SRC_FILE} < ${DATA_DIR}/$(basename $file) > ${DATA_DIR}/$(basename $file).tok

  wc -l ${DATA_DIR}/$(basename $file).tok

  if [[ "$file" = *"mt"* ]]; then
    python tokenisemt.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
  elif [[ "$file" = *"ic"* ]]; then
    python tokeniseic.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode

  else
    cat ${DATA_DIR}/$(basename $file).tok \
    | $TOKENIZER -l $SRC_FILE \
    > ${DATA_DIR}/$(basename $file)
  fi

  wc -l ${DATA_DIR}/$(basename $file)
  rm ${DATA_DIR}/$(basename $file).tok
  echo ${DATA_DIR}/$(basename $file)
done

for file in ${DATA_DIR}/*.${TGT_FILE}; do
  wc -l ${DATA_DIR}/$(basename $file)
  $TRUECASER --model truecase-model.${TGT_FILE} < ${DATA_DIR}/$(basename $file) > ${DATA_DIR}/$(basename $file).tok

  wc -l ${DATA_DIR}/$(basename $file).tok
  if [[ "$file" = *"mt"* ]]; then
    python tokenisemt.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
  elif [[ "$file" = *"ic"* ]]; then
    python tokeniseic.py --src ${DATA_DIR}/$(basename $file).tok --tgt ${DATA_DIR}/$(basename $file) --encode
  else
    cat ${DATA_DIR}/$(basename $file).tok \
    | $TOKENIZER -l $SRC_FILE \
    > ${DATA_DIR}/$(basename $file)
  fi
  

  rm ${DATA_DIR}/$(basename $file).tok
  wc -l ${DATA_DIR}/$(basename $file)
  echo ${DATA_DIR}/$(basename $file)
done

echo "applied truecasing and tokenized train/dev/test"








echo "applying BPE next:"
python sp_tools.py \
  --encode \
  --src ${SRC_FILE} \
  --tgt ${TGT_FILE} \
  --data_dir ${DATA_DIR}/ \
  --data_out ${DATA_OUT}/ \
  --model_dir ${MODELS}/ \
  --vocab_size 32000

echo "################ Done tokenizing ################"

# Use rule.py to generate the noisy data
python secoco/rule.py \
  --data_out ${DATA_OUT}/ \
  --data_dir ${DATA_DIR}/ \
  --src ${SRC_FILE} 


# Use export_for_task.py to extract the edits
python secoco/export_for_task.py \
  --data_out ${DATA_OUT}/ \
  --data_dir ${DATA_DIR}/ \
  --src ${SRC_FILE} 


rm -f $DATA_OUT/train.bpe.tag.$SRC_FILE
rm -f $DATA_OUT/dev.bpe.tag.$SRC_FILE
rm -f $DATA_OUT/test.bpe.tag.$SRC_FILE
echo "################ Encoding Data ################"

 python ./secoco/fairseq/fairseq_cli/preprocess_ende.py  --source-lang $SRC_FILE --target-lang $TGT_FILE \
	--trainpref $DATA_OUT/train.bpe \
    --validpref $DATA_OUT/dev.bpe \
    --destdir $DATA_OUT \
    --srcdict $MODELS/fairseq.dict \
    --tgtdict $MODELS/fairseq.dict \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 4 \
    --testpref $DATA_OUT/test.bpe

echo "################ Done encoding ################"