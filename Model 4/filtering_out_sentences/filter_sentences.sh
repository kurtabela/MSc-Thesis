#!/bin/bash
# let's set the following defaults (can be overriden on commandline):
#SBATCH --job-name filter_out_sentences
#SBATCH --partition A100
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks=40
#SBATCH --mem=920G
#SBATCH --time=2-00:00




#rm /netscratch/abela/transformerbaseline/test_dataset/data/prediction*
#rm -r checkpoints

train_en='../data_test/original_data/en_mt/train.en'
train_mt='../data_test/original_data/en_mt/train.mt'

total_lines=$(wc -l < $train_en )
((lines_per_file = (total_lines ) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${train_en} '../data_test/original_data/en_mt/train'



total_lines=$(wc -l < $train_mt )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose ${train_mt} '../data_test/original_data/en_mt/train'




dev_en='../data_test/original_data/en_mt/dev.en'
total_lines=$(wc -l < $dev_en )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${dev_en} '../data_test/original_data/en_mt/dev'





dev_mt='../data_test/original_data/en_mt/dev.mt'
total_lines=$(wc -l < $dev_mt )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose ${dev_mt} '../data_test/original_data/en_mt/dev'


test_en='../data_test/original_data/en_mt/test.en'
test_mt='../data_test/original_data/en_mt/test.mt'
total_lines=$(wc -l < $test_en )
((lines_per_file = (total_lines ) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".en" --verbose ${test_en} '../data_test/original_data/en_mt/test'



total_lines=$(wc -l < $test_mt )
((lines_per_file = (total_lines) / ${SLURM_NTASKS}))

split --lines=${lines_per_file} --numeric-suffixes=1 --suffix-length=2 --additional-suffix=".mt" --verbose ${test_mt} '../data_test/original_data/en_mt/test'




# First we do the train set

((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_mt/train${nOfTasks}"
  inp2="../filtered_data/en_mt/train${nOfTasks}"
  srun  --exclusive -N1 -n1 --threads-per-core=1 --mem=23G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"  --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_mt.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done


# Dev set



((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_mt/dev${nOfTasks}"
  inp2="../filtered_data/en_mt/dev${nOfTasks}"
  srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=30G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_mt.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done


#
#inp1='../data_test/original_data/en_mt/dev'
#inp2="../filtered_data/en_mt/train${nOfTasks}"
#srun  -K --gpus=1 --mem-per-cpu=15G -N1 --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`",/home/abela/thesis:/home/abela/thesis  --task-prolog=install.sh ipython scoring_sentenced.py ${inp1} ${inp2} &
#  echo ${nOfTasks}


# Test set
((tasksminus1 = (${SLURM_NTASKS}-1)))
echo ${tasksminus1}
for nOfTasks in $(seq -w 01 ${tasksminus1})
do
  inp1="../data_test/original_data/en_mt/test${nOfTasks}"
  inp2="../filtered_data/en_mt/test${nOfTasks}"
  srun   --exclusive -N1 -n1 --threads-per-core=1 --mem=23G --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh   --container-workdir="`pwd`"   --container-mounts=/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"  --task-prolog=install.sh ipython scoring_sentenced_mt.py ${inp1} ${inp2} &
  echo ${nOfTasks}
done

wait
