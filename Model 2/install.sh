#!/bin/bash
pip install psutil
pip install numpy
pip install sacrebleu
pip install sentencepiece
pip install -q tfds-nightly
pip uninstall -y tensorflow
pip install tensorflow
pip install tensorflow_datasets
pip install tokenizer
pip install tensor2tensor

pip install tensorflow-text
pip install --user --upgrade tensorflow-model-optimization
pip install tensorflow