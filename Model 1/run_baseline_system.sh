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

cd fairseq
pip install -e .
cd ..

echo "################ Starting training ################"

echo $EPOCHS
${FAIRSEQ_TRAIN} \
    $DATA_OUT \
    --source-lang $SRC_FILE --target-lang $TGT_FILE \
    --batch-size 64 --arch transformer --share-all-embeddings \
    --encoder-layers 5 --decoder-layers 5 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.4  --attention-dropout 0.2 --relu-dropout 0.2 \
    --weight-decay 0.0001 \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 \
    --max-tokens 1700 \
    --update-freq 4 \
    --max-epoch $EPOCHS --save-interval 1 \
    --tensorboard-logdir $LOGS \
    --save-dir $CHECKPOINTS \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints

echo "################ Done training, starting evaluation ################"

${FAIRSEQ_GENERATE} \
      $DATA_OUT   \
    --source-lang $SRC_FILE --target-lang $TGT_FILE \
    --path $CHECKPOINTS/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 \
    --gen-subset test --batch-size 32 --max-tokens-valid 2048 --skip-invalid-size-inputs-valid-test  > $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.out

echo "################ Done fairseq-generate ################"

cat $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.out | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp
cat $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.out | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[ro_RO\]//g'  > $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref
cp $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.tokenized

echo "################ SP TOOLS################"

python sp_tools.py \
  --decode \
  --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.tokenized \
  --tgt $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp \
  --model_dir ${MODELS}/

cp $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.tokenized
python sp_tools.py \
  --decode \
  --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.tokenized \
  --tgt $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref \
  --model_dir ${MODELS}/






if [[ ${TGT_FILE} = "mt" ]]; then


    python tokenisemt.py \
        --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref \
        --tgt  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.detokenized \
        --decode
        
elif [[ ${TGT_FILE} = "ic" ]]; then

    python tokeniseic.py \
        --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref \
        --tgt  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.detokenized \
        --decode
else

    cat $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref \
    | $DETOKENIZER -l $TGT_FILE \
    > $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.detokenized
fi





if [[ ${TGT_FILE} = "mt" ]]; then

     python tokenisemt.py \
        --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp \
        --tgt  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.detokenized \
        --decode




        
elif [[ ${TGT_FILE} = "ic" ]]; then
     python tokeniseic.py \
        --src $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp \
        --tgt  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.detokenized \
        --decode



else

    cat $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp \
    | $DETOKENIZER -l $TGT_FILE \
    > $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.detokenized

fi




echo "################ Done decoding ################"
echo  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.detokenized  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.detokenized

${EVALUATE} \
  --system_output $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.hyp.detokenized \
  --gold_reference  $TRANSLATIONS/${SRC_FILE}_${TGT_FILE}.bpe.ref.detokenized


