#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name secoco
#SBATCH --partition A100
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem=70G
#SBATCH --time=3-00:00




#rm -rf results/
#rm -rf data/
#cp -rf original_data/ data/

srun  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.06-py3.sqsh --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/home/abela/thesis:/home/abela/thesis  --task-prolog=install.sh run_baseline_system.sh ic en  data results/ 3 -CLI
