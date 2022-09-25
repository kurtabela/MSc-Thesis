#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name pruning_bleu
#SBATCH --partition RTXA6000
#SBATCH --mem=150G
#SBATCH --time=01-00:00
#rm /netscratch/abela/pruning/test_dataset/data/prediction*
#rm -r checkpoints


#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/test_dataset/data/detokenizer.perl -l mt < /netscratch/abela/transformer_baseline/test_dataset/data/predmttions.mt > test_dataset/data/predmttions.mt.detok
srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

