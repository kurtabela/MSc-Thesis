#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multisense_baseline
#SBATCH --partition RTX3090
#SBATCH --gpus=3
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=2
#SBATCH --time=3-00:00
#SBATCH --nodes=1
#rm -r test_dataset/translator
#rm -r checkpoints
ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n


#srun  -K --gpus=1 --mem=32G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install_train_baseline.sh ipython subwords_tokenizer.py
srun  -K --gpus=3 --mem=48G --cpus-per-gpu=2 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython train_baseline.py
#srun  -K --gpus=1 --mem=32G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython test_set.py
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh test_dataset/data/detokenizer.perl -l mt < test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --mem=32G --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate.py
