#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name baseline_preprocessing
#SBATCH --partition RTXA6000
#SBATCH --mem=60G
#rm -rf results/


#rm -rf data/
#cp -rf original_data/ data/

srun  --mem=60G --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh preprocessing.sh mt en  data/ results/ 10 -CLI
