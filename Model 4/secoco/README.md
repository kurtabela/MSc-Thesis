# Secoco
Secoco: Self-Correcting Encoding for Neural Machine Translation

## How to generate data

All scripts for generating training/test data of editing are saved in `generate_data_xx` folder.

Using `rule.py` to generate noisy data with pre-defined rules, and `export_for_task.py` to extract key edits and positions for futher processing. 

We can also use `edit.py` to get edits when having reference.

These generated edits are then used to generate binary files of sequence tagging signals for training, which is hardcoded in `fairseq/fairseq/binarizer_xx.py`.


## Training
Most of the modifications are in `fairseq/fairseq/models/transformer.py` and `fairseq/fairseq/criterions/label_smoothed_cross_entropy.py`

## Infer
Using `fairseq/fairseq_cli/interactive_preedit.py` to do interactive editing before translating. 