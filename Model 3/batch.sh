#!/bin/bash
#let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name pruning
#SBATCH --partition batch
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=70G
#SBATCH --nodes=1
#SBATCH --time=7-00:00

#rm -r test_dataset/translator
#rm -r checkpoints

#srun  -K --gpus=3 --mem=60G --cpus-per-gpu=2 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython subwords_tokenizer.py
srun  -K --gpus=1 --cpus-per-gpu=2 --mem=70G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython transformer.py
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython test_set.py
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh test_dataset/data/detokenizer.perl -l mt < test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate.py
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh comet-score -s test_dataset/data/test.en -t test_dataset/data/predictions.mt.detok -r test_dataset/data/test.mt
