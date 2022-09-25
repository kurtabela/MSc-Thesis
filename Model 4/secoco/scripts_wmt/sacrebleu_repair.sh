#!/bin/bash
TOKENIZEROOT=mosesdecoder/scripts/tokenizer

DATABIN=$1
MODEL=$2
MODEL2=$3
INPUT=$4
OUTPUT=$5
REF=$6

echo repairing $INPUT from $SRCLANG to $TRGLANG and storing to $OUTPUT using $MODEL and $DATABIN

cat $INPUT \
| fairseq-interactive $DATABIN --path $MODEL -s src -t trg --beam 5 --max-tokens 3000 --buffer-size 3000 \
| grep ^H- | cut -f 3- \
| fairseq-interactive $DATABIN --path $MODEL2 -s en -t de --beam 5 --remove-bpe --max-tokens 3000 --buffer-size 3000 \
| grep ^H- | cut -f 3- \
| perl $TOKENIZEROOT/detokenizer.perl -l de -q > $OUTPUT

if [ ! $REF ]; then
    echo NO REFERENCE!
else
    cat $OUTPUT | sacrebleu -l en-de $REF
fi