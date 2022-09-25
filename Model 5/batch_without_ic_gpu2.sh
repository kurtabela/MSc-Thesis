#!/bin/bash
#let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multisense_create_updated_ic
#SBATCH --partition batch
#SBATCH --mem=125G
#SBATCH --gpus=1

#rm -r test_dataset/translator
#rm -r checkpoints

#srun  --mem=100G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython subwords_tokenizer.py
srun  -K --gpus=1 --mem=125G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython create_updated_data_en_ic.py
#srun  -K --mem=100G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython train_baseline-ic.py
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh test_dataset/data/detokenizer.perl -l mt < test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --gpus=1 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh python evaluate.py
