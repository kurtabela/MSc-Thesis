#!/bin/bash


# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then


  pip install sacrebleu
  pip install sentencepiece
  pip install -q tfds-nightly
  pip install tensorflow_datasets
  pip install tokenizer
  pip install tensor2tensor
  pip install tensorflow
  pip install --user --upgrade tensorflow-model-optimization  
  pip install tensorflow_text
  echo "DONE"
 
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi


# This runs your wrapped command
"$@"
