#!/bin/bash
#let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multisense_main
#SBATCH --partition A100
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --time=0-20:00

ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n
#rm -r ic_en_dataset/translator
#rm -r checkpoints

#srun  -K --gpus=1 --mem=32G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython subwords_tokenizer-ic.py
#srun  -K --gpus=1 --mem=200G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh    --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython create_updated_data_en_ic.py
srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh    --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython create_updated_data_step3_ic.py

srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython transformer-ic.py
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ic_en_dataset/data/detokenizer.perl -l ic < ic_en_dataset/data/predictions.ic > ic_en_dataset/data/predictions.ic.detok
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate-ic.py
