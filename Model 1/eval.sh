#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name baseline_eval
#SBATCH --partition RTXA6000
#SBATCH --mem=200G 
#SBATCH --time=1-00:00

#rm -rf results/


#rm -rf data/
#cp -rf original_data/ data/

srun  --mem=200G  --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.08-py3.sqsh  --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"  --task-prolog=install.sh run_baseline_system_eval.sh mt en data/ results/ 10 -CLI
