#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name monomt
#SBATCH --partition batch
#SBATCH --mem=64G 

echo "tf?"
srun  -K --mem=64G  --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh to_predict_mono_data.sh en mt mono_mt/ mono_mt_results/ 10 -CLI
