#!/bin/bash
pip install -r requirements.txt
pip install tensorboardX
pip install mosestokenizer
pip install sentencepiece
pip install sentence_splitter
pip uninstall -y torch
cd fairseq
pip install fairseq
cd ..
pip install tokenizer
pip install --upgrade numpy
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
