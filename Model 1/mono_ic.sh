#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name baseline1
#SBATCH --partition RTX2080Ti
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G 

echo "tf?"
srun  -K --mem=64G  --gpus=1 --cpus-per-gpu=2 --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh to_predict_mono_data_ic.sh en ic mono_ic/ mono_ic_results/ 10 -CLI
