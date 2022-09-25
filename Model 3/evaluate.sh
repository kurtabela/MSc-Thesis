#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name pruning_eval
#SBATCH --partition A100
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks=38
#SBATCH --nodes=2
#SBATCH --mem=450G
#SBATCH --time=03-00:00
#rm /netscratch/abela/pruning/test_dataset/data/prediction*
#rm -r checkpoints



echo "HELLO"
pip list


total_lines=$(wc -l < /netscratch/abela/pruning/test_dataset/data/test.en )
((lines_per_file = (total_lines ) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose /netscratch/abela/pruning/test_dataset/data/test.en /netscratch/abela/pruning/test_dataset/data/test



total_lines=$(wc -l < /netscratch/abela/pruning/test_dataset/data/test.mt )
((lines_per_file = (total_lines) / (${SLURM_NTASKS}-1)))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose /netscratch/abela/pruning/test_dataset/data/test.mt /netscratch/abela/pruning/test_dataset/data/test


((tasksminus1 = (${SLURM_NTASKS})))
echo ${tasksminus1}
for nOfTasks in $(seq 1 ${tasksminus1})
do
   srun   -N1 -n1 --threads-per-core=1 --mem=22G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`" --task-prolog=install.sh ipython test_set.py ${nOfTasks} &
  echo ${nOfTasks}
done
wait
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro --task-prolog=install.sh /netscratch/abela/transformer_baseline/test_dataset/data/detokenizer.perl -l mt < /netscratch/abela/transformer_baseline/test_dataset/data/predmttions.mt > test_dataset/data/predmttions.mt.detok
#srun  -K --container-image=/netscratch/enroot/nvcr.io_nvidia_tensorflow_21.12-tf2-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER/pruning:"`pwd`",/ds:/ds:ro  --task-prolog=install.sh python evaluate.py

