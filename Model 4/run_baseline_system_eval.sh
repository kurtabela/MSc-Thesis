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

MODELS=${SAVE_DIR}/models/${TGT_FILE}_${SRC_FILE}
CHECKPOINTS=${SAVE_DIR}/checkpoints/${TGT_FILE}_${SRC_FILE}
LOGS=${SAVE_DIR}/logs/${TGT_FILE}_${SRC_FILE}
TRANSLATIONS=${SAVE_DIR}/translations/${TGT_FILE}_${SRC_FILE}
DATA_OUT=${SAVE_DIR}/data_out/${TGT_FILE}_${SRC_FILE}
ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n


cd secoco/fairseq 
pip install . 
cd ../../
#pw


#cd secoco
#python setup.py build_ext --inplace
#pip install -e fairseq --log piplog.txt
#cd ../


pip list


echo translating $DATA_DIR/test-short.${SRC_FILE} from $SRC_FILE to $TGT_FILE and storing to  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp using $CHECKPOINTS/checkpoint_best.pt and $DATA_OUT

cat $DATA_DIR/test-short.${SRC_FILE} \
|   python ./secoco/fairseq/fairseq_cli/interactive.py  $DATA_OUT --input=$DATA_DIR/test.${SRC_FILE} --path $CHECKPOINTS/checkpoint_best.pt --sentencepiece-vocab ${MODELS}/sentencepiece.bpe.model -s $SRC_FILE -t $TGT_FILE --bpe sentencepiece --beam 5 --remove-bpe True --max-tokens 3000 --buffer-size 3000 --max-source-positions 128 --max-target-positions 128 --skip-invalid-size-inputs-valid-test \
>  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out




echo "################ Done fairseq-generate ################"

cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp


cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.out | grep -P "^S" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp.src

python sp_tools.py \
  --decode \
  --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp.src \
  --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
  --model_dir ${MODELS}/




if [[ ${SRC_FILE} = "en" ]]; then
    cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
    | $DETOKENIZER -l $SRC_FILE \
    > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src
#    echo "Detokenizing..."
#    cat $DATA_DIR/test.${SRC_FILE} \
#    | $DETOKENIZER -l $SRC_FILE \
#    > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.originalSrc
#    
    echo "Done"

elif [[ ${SRC_FILE} = "ic" ]]; then
    python tokeniseic.py \
        --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
        --tgt  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src \
        --decode
    python tokeniseic.py \
        --src $DATA_DIR/test.${SRC_FILE} \
        --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.originalSrc \
        --decode
else
    python tokenisemt.py \
        --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.temp2.src \
        --tgt  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src \
        --decode
    python tokenisemt.py \
        --src $DATA_DIR/test.${SRC_FILE} \
        --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.originalSrc \
        --decode
fi


echo "DOING CREATE REF"
echo  $DATA_DIR/test.${SRC_FILE} $DATA_DIR/test.${TGT_FILE} $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.ref
python createRefFile.py --originalSrc $DATA_DIR/test.${SRC_FILE}  --originalTgt $DATA_DIR/test.${TGT_FILE} --sourceHyp $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.src --target $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.ref
echo "DONE"


cp $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.tokenized

echo "################ SP TOOLS################"

python sp_tools.py \
  --decode \
  --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.tokenized \
  --tgt $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp \
  --model_dir ${MODELS}/


echo ${TGT_FILE}
if [[ ${TGT_FILE} = "mt" ]]; then
    
    cat $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp \
    | $DETOKENIZER -l $TGT_FILE \
    > $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.detokenized

else

    python tokenisemt.py \
        --src $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp \
        --tgt  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.detokenized \
        --decode
fi

wc -l $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.detokenized
wc -l $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.ref
echo "################ Done decoding ################"

${EVALUATE} \
  --system_output $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.hyp.detokenized \
  --gold_reference  $TRANSLATIONS/${TGT_FILE}_${SRC_FILE}.bpe.ref


