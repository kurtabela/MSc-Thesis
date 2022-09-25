#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multi_step3
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=23
#SBATCH --mem=400G
#rm /netscratch/abela/transformerbaseline/test_dataset/data/prediction*
#rm -r checkpoints



echo "HELLO"
pip list


total_lines=$(wc -l < /netscratch/abela/multi-sense-words/dataset_without_masks/results.mt )
((lines_per_file = (total_lines ) / (${SLURM_NTASKS}-1)))


split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose /netscratch/abela/multi-sense-words/dataset_without_masks/results.mt /netscratch/abela/multi-sense-words/dataset_without_masks/results


((tasksminus1 = (${SLURM_NTASKS})))
echo ${tasksminus1}
for nOfTasks in $(seq 1 ${tasksminus1})
do
   srun  --mem=25G -N1 --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython create_updated_data_step3.py ${nOfTasks} &
  echo ${nOfTasks}
done
wait
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/test_dataset/data/detokenizer.perl -l mt < /netscratch/abela/multi-sense-words/dataset_without_masks/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

