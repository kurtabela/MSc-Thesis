#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multisense_baseline_ic
#SBATCH --partition A100
#SBATCH --mem=48G
#SBATCH --time=2-00:00

#SBATCH --gpus=3

#SBATCH --nodes=1
#rm -r test_dataset/translator
#rm -r checkpoints
ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n

#srun  -K --mem=48G --gpus=3 --cpus-per-gpu=2 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install_train_baseline.sh ipython subwords_tokenizer-ic.py
srun  -K --mem=48G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython train_baseline-ic.py
#srun  -K --gpus=1 --mem=32G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython test_set-ic.py
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ic_en_dataset/data/detokenizer.perl -l mt < ic_en_dataset/data/predictions.ic > ic_en_dataset/data/predictions.ic.detok
#srun  -K --mem=32G --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate-ic.py

