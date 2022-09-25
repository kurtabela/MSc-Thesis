#!/bin/bash
TOKENIZEROOT=../basic_tools/mosesdecoder/scripts/tokenizer

DATABIN=$1
MODEL=$2
SRCLANG=$3
TRGLANG=$4
INPUT=$5
OUTPUT=$6
REF=$7

echo translating $INPUT from $SRCLANG to $TRGLANG and storing to $OUTPUT using $MODEL and $DATABIN

cat $INPUT \
| fairseq-interactive $DATABIN --path $MODEL -s $SRCLANG -t $TRGLANG --beam 5 --remove-bpe --max-tokens 3000 --buffer-size 3000 \
    --max-source-positions 128 --max-target-positions 128 --skip-invalid-size-inputs-valid-test \
| grep ^H- | cut -f 3- \
| perl $TOKENIZEROOT/detokenizer.perl -l $TRGLANG -q > $OUTPUT

if [ ! $REF ]; then
    echo NO REFERENCE!
else
    cat $OUTPUT | sacrebleu -l en-de $REF
fi
