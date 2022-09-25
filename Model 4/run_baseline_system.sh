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



cd secoco/ 
#python setup.py build_ext --inplace
pip install -e fairseq --log piplog.txt || true
cd ../



echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo $LOCAL_RANK
echo $SLURM_PROCID
export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
#export MKL_SERVICE_FORCE_INTEL=1
#pip list
#pwd

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo "################ Starting training 2 ################"

python ./secoco/fairseq/fairseq_cli/train.py    $DATA_OUT \
    --source-lang $SRC_FILE --target-lang $TGT_FILE \
    --arch transformer --share-all-embeddings \
    --encoder-layers 5 --decoder-layers 5 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
    --weight-decay 0.0001 \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 \
    --max-tokens 80 --max-tokens-valid 1024 --fp16  \
    --update-freq 20 \
    --max-epoch $EPOCHS --save-interval 1 \
    --save-dir $CHECKPOINTS \
    --no-progress-bar --skip-invalid-size-inputs-valid-test --save-interval-updates 10000


echo "################ Done training, starting evaluation ################"


#echo $DATA_DIR/test.en
#echo $DATA_OUT
#echo $CHECKPOINTS/checkpoint_best.pt 
#
#cat $DATA_DIR/test.en \
#| fairseq-interactive $DATA_OUT --path $CHECKPOINTS/checkpoint_best.pt  -s $SRC_FILE -t $TGT_FILE --beam 5 --remove-bpe --max-tokens 3000 --buffer-size 3000 \
#    --max-source-positions 128 --max-target-positions 128 --skip-invalid-size-inputs-valid-test 

#

