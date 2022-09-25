#!/bin/bash
#let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multisense_main
#SBATCH --partition A100
#SBATCH --gpus=3
#SBATCH --mem=150G
#SBATCH --time=03-00:00
#SBATCH --nodes=1
ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n

#rm -r test_dataset/translator
#rm -r checkpoints

#srun  -K --mem=120G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython subwords_tokenizer.py
#srun  -K --gpus=1 --mem=150G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh    --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython create_updated_data.py
#srun  -K --gpus=1 --mem=150G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh    --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython create_updated_data_step3.py
srun   --mem=150G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython transformer.py
#srun  -K --gpus=0 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh test_dataset/data/detokenizer.perl -l mt < test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --gpus=0 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate.py
