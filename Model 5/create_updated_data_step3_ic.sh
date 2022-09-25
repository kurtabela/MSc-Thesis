#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name multi_step3_ic
#SBATCH --partition A100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=23
#SBATCH --mem=400G
#SBATCH --time=03-00:00
#rm /netscratch/abela/transformerbaseline/test_dataset/data/prediction*
#rm -r checkpoints



echo "HELLO"
pip list


total_lines=$(wc -l < /netscratch/abela/multi-sense-words/dataset_without_mask_ic/results.ic )
((lines_per_file = (total_lines ) / (${SLURM_NTASKS}-1)))


split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".ic" --verbose /netscratch/abela/multi-sense-words/dataset_without_masks_ic/results.ic /netscratch/abela/multi-sense-words/dataset_without_masks_ic/results


((tasksminus1 = (${SLURM_NTASKS})))
echo ${tasksminus1}
for nOfTasks in $(seq 1 ${tasksminus1})
do
   srun  --mem=22G -N1 --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython create_updated_data_step3_ic.py ${nOfTasks} &
  echo ${nOfTasks}
done
wait
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/test_dataset/data/detokenizer.perl -l mt < /netscratch/abela/multi-sense-words/dataset_without_masks/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

