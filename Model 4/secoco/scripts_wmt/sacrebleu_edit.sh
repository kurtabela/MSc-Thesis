#!/bin/bash
TOKENIZEROOT=mosesdecoder/scripts/tokenizer

DATABIN=$1
MODEL=$2
SRCLANG=$3
TRGLANG=$4
INPUT=$5
OUTPUT=$6
REF=$7

echo translating $INPUT from $SRCLANG to $TRGLANG and storing to $OUTPUT using $MODEL and $DATABIN

cat $INPUT \
| python ../fairseq/fairseq_cli/interactive_preedit.py $DATABIN --path $MODEL -s $SRCLANG -t $TRGLANG --beam 5 --remove-bpe --max-tokens 3000 --buffer-size 3000 \
| grep ^H- | cut -f 3- \
| perl $TOKENIZEROOT/detokenizer.perl -l $TRGLANG -q > $OUTPUT

if [ ! $REF ]; then
    echo NO REFERENCE!
else
    cat $OUTPUT | sacrebleu -l en-de $REF
fi