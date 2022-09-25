#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name baseline1
#SBATCH --partition A100
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --time=1-00:00

#rm -rf results/


#rm -rf data/
#cp -rf original_data/ data/

srun  --mem=150G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh run_baseline_system.sh mt en data/ results/ 10 -CLI
