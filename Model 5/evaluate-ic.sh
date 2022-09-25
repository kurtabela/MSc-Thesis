#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name transformer_eval
#SBATCH --partition A100
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks=32
#SBATCH --mem=860G
#SBATCH --time=3-00:00

#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/prediction*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/test0*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/test1*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/test2*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/test3*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/test4*
#rm /netscratch/abela/transformerbaseline/ic_en_dataset/data/tooBigPrediction*

#rm -r checkpoints



echo "HELLO"
pip list


total_lines=$(wc -l < /netscratch/abela/multi-sense-words/ic_en_dataset/data/test.en )
((lines_per_file = (total_lines ) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose /netscratch/abela/multi-sense-words/ic_en_dataset/data/test.en /netscratch/abela/multi-sense-words/ic_en_dataset/data/test



total_lines=$(wc -l < /netscratch/abela/multi-sense-words/ic_en_dataset/data/test.ic )
((lines_per_file = (total_lines) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".ic" --verbose /netscratch/abela/multi-sense-words/ic_en_dataset/data/test.ic /netscratch/abela/multi-sense-words/ic_en_dataset/data/test


((tasksminus1 = (${SLURM_NTASKS})))
echo ${tasksminus1}
for nOfTasks in $(seq 1 ${tasksminus1})
do
   srun  -N1 -n1 --threads-per-core=1 --mem=25G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython test_set-ic.py ${nOfTasks} &
  echo ${nOfTasks}
done
wait
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/ic_en_dataset/data/detokenizer.perl -l ic < /netscratch/abela/transformer_baseline/ic_en_dataset/data/predictions.ic > ic_en_dataset/data/predictions.ic.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

