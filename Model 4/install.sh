#!/bin/bash
#
#pip install torch==1.6.0 torchvision==0.7.0
pip install tensorboardX
pip install mosestokenizer
pip install sentencepiece
pip install sentence_splitter
pip install tensorflow_text

pip install tokenizer
pip install joblib

cd secoco/ 
#python setup.py build_ext --inplace
pip install -e fairseq --log piplog.txt || true
cd ../

echo CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
echo $LOCAL_RANK
echo $SLURM_PROCID
#export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
