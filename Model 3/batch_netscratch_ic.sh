#!/bin/bash
#let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name pruning_ic
#SBATCH --partition A100
#SBATCH --gpus=3
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --time=3-00:00

ulimit -s
ulimit -n
ulimit -n 131071
ulimit -s 131070
ulimit -s
ulimit -n

#rm -r test_dataset/translator
#rm -r checkpoints

#srun  -K --gpus=6 --mem=60G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"    --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh ipython subwords_tokenizer-ic.py
srun  -K --gpus=3 --mem=60G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh ipython transformer-ic.py
#srun  -K --gpus=1 --mem=82G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh ipython test_set.py
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro --task-prolog=install.sh test_dataset/data/detokenizer.perl -l mt < test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh comet-score -s test_dataset/data/test.en -t test_dataset/data/predictions.mt.detok -r test_dataset/data/test.mt
