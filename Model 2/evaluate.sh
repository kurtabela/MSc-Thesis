#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name transformer_eval
#SBATCH --partition A100
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks=30
#SBATCH --time=01-00:00
#SBATCH --mem=850G


#rm /netscratch/abela/transformerbaseline/test_dataset/data/prediction*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/test0*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/test1*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/test2*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/test3*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/test4*
#rm /netscratch/abela/transformerbaseline/test_dataset/data/tooBigPredictions*
#rm -r checkpoints



echo "HELLO"
pip list


total_lines=$(wc -l < /netscratch/abela/transformerbaseline/test_dataset/data/test.en )
((lines_per_file = (total_lines ) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose /netscratch/abela/transformerbaseline/test_dataset/data/test.en /netscratch/abela/transformerbaseline/test_dataset/data/test



total_lines=$(wc -l < /netscratch/abela/transformerbaseline/test_dataset/data/test.mt )
((lines_per_file = (total_lines) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose /netscratch/abela/transformerbaseline/test_dataset/data/test.mt /netscratch/abela/transformerbaseline/test_dataset/data/test

#
#srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=200G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython test_set.py 1

((tasksminus1 = (${SLURM_NTASKS})))
echo ${tasksminus1}
for nOfTasks in $(seq 1 ${tasksminus1})
do
  srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=26G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython test_set.py ${nOfTasks} &
  echo ${nOfTasks}
done
wait
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/test_dataset/data/detokenizer.perl -l mt < /netscratch/abela/transformer_baseline/test_dataset/data/predictions.mt > test_dataset/data/predictions.mt.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/transformerbaseline:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

