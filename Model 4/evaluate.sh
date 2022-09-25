#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name secoco_eval
#SBATCH --partition A100
#SBATCH --cpus-per-task=50
#SBATCH --mem=100G


#rm -rf results/
#rm -rf data/
#cp -rf original_data/ data/

srun  -K  --mem=100G  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.06-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/home/abela/thesis:/home/abela/thesis  --task-prolog=install.sh run_baseline_system_eval.sh mt en data results/ 10 -CLI

#--container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh
