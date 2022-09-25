#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name filter_out_sentences_ic
#SBATCH --partition A100
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks=10
#SBATCH --mem=320G
#SBATCH --time=2-00:00




#rm /netscratch/abela/transformerbaseline/test_dataset/data/prediction*
#rm -r checkpoints

train_en='../data_test/original_data/en_ic/train.en'
train_ic='../data_test/original_data/en_ic/train.ic'

total_lines=$(wc -l < $train_en )
((lines_per_file = (total_lines ) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${train_en} '../data_test/original_data/en_ic/train'



total_lines=$(wc -l < $train_ic )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".ic" --verbose ${train_ic} '../data_test/original_data/en_ic/train'




dev_en='../data_test/original_data/en_ic/dev.en'
total_lines=$(wc -l < $dev_en )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${dev_en} '../data_test/original_data/en_ic/dev'





dev_ic='../data_test/original_data/en_ic/dev.ic'
total_lines=$(wc -l < $dev_ic )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".ic" --verbose ${dev_ic} '../data_test/original_data/en_ic/dev'


test_en='../data_test/original_data/en_ic/test.en'
test_ic='../data_test/original_data/en_ic/test.ic'
total_lines=$(wc -l < $test_en )
((lines_per_file = (total_lines ) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${test_en} '../data_test/original_data/en_ic/test'



total_lines=$(wc -l < $test_ic )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".ic" --verbose ${test_ic} '../data_test/original_data/en_ic/test'




# First we do the train set

((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_ic/train${nOfTasks}"
  inp2="../filtered_data/en_ic/train${nOfTasks}"
  srun  --exclusive -N1 -n1 --threads-per-core=1 --mem=30G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"  --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_en_ic.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done


# Dev set



((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_ic/dev${nOfTasks}"
  inp2="../filtered_data/en_ic/dev${nOfTasks}"
  srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=30G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_en_ic.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done


#
#inp1='../data_test/original_data/en_ic/dev'
#inp2="../filtered_data/en_ic/train${nOfTasks}"
#srun  -K --gpus=1 --mem-per-cpu=15G -N1 --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/home/abela/thesis:/home/abela/thesis  --task-prolog=install.sh ipython scoring_sentenced.py ${inp1} ${inp2} &
#  echo ${nOfTasks}


# Test set
((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_ic/test${nOfTasks}"
  inp2="../filtered_data/en_ic/test${nOfTasks}"
  srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=30G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_en_ic.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done

wait
