# -*- coding: utf-8 -*-
# +

get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install tensorflow_text')
get_ipython().system('pip install tensorflow-model-optimization')
get_ipython().system('pip install -U "tensorflow-text==2.8.*"')
get_ipython().system('pip install sentence_splitter')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install -q tfds-nightly tensorflow matplotlib')


import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time
from subprocess import PIPE, Popen
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

import test_dataset

import tempfile
# get_ipython().run_line_magic('cd', 'test_dataset')
os.chdir("test_dataset")
os.system("tfds build")


"""## Download the dataset

Fetch the Maltese/English translation dataset from [tfds](https://tensorflow.org/datasets):
"""



# -


examples, metadata = tfds.load(name="test_dataset", with_info=True,
                               as_supervised=True, data_dir="test_dataset")
train_examples, val_examples = examples['train'], examples['validation']  

for en, mt in train_examples.take(1):
    print("English: ", en.numpy().decode('utf-8'))
    print("Maltese:   ", mt.numpy().decode('utf-8'))

train_en = train_examples.map(lambda en, mt: en)
train_mt = train_examples.map(lambda en, mt: mt)

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

# +
bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = 32000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)
# -

# %%time
mt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_mt.batch(1000).prefetch(2),
    **bert_vocab_args
)

print(mt_vocab[:10])
print(mt_vocab[100:110])
print(mt_vocab[1000:1010])
print(mt_vocab[-10:])


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


# %%time
en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
)


print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])

write_vocab_file('en_vocab.txt', en_vocab)
write_vocab_file('mt_vocab.txt', mt_vocab)
# !ls *.txt

mt_tokenizer = text.BertTokenizer('mt_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)

for mt_examples, en_examples in train_examples.batch(3).take(1):
    for ex in en_examples:
        print(ex.numpy())

# +
# Tokenize the examples -> (batch, word, word-piece)
token_batch = en_tokenizer.tokenize(en_examples)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2,-1)

for ex in token_batch.to_list():
    print(ex)
# -

# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(en_vocab, token_batch)
# Join with spaces.
tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1)

words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)

# +
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], START)
    ends = tf.fill([count,1], END)
    return tf.concat([starts, ragged, ends], axis=1)


# -

words = en_tokenizer.detokenize(add_start_end(token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


en_examples.numpy()

token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2,-1)
words = en_tokenizer.detokenize(token_batch)
words

cleanup_text(reserved_tokens, words).numpy()


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:   

        # Include a tokenize signature for a batch of strings. 
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2,-1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


tokenizers = tf.Module()
tokenizers.mt = CustomTokenizer(reserved_tokens, 'mt_vocab.txt')
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')

model_name = 'ted_hrlr_translate_mt_en_converter'
tf.saved_model.save(tokenizers, model_name)

reloaded_tokenizers = tf.saved_model.load(model_name)
reloaded_tokenizers.en.get_vocab_size().numpy()

tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
tokens.numpy()

# +
round_trip = reloaded_tokenizers.en.detokenize(tokens)

print(round_trip.numpy()[0].decode('utf-8'))
# -

mt_lookup = tf.lookup.StaticVocabularyTable(
    num_oov_buckets=1,
    initializer=tf.lookup.TextFileInitializer(
        filename='mt_vocab.txt',
        key_dtype=tf.string,
        key_index = tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype = tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER)) 
mt_tokenizer = text.BertTokenizer(mt_lookup)

mt_lookup.lookup(tf.constant(['Aw', 'dinja', 'illum', 'temp', 'ikrah']))

mt_lookup = tf.lookup.StaticVocabularyTable(
    num_oov_buckets=1,
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=mt_vocab,
        values=tf.range(len(mt_vocab), dtype=tf.int64))) 
mt_tokenizer = text.BertTokenizer(mt_lookup)

# +
tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
tokens.numpy()

round_trip = reloaded_tokenizers.en.detokenize(tokens)

print(round_trip.numpy()[0].decode('utf-8'))
# +
tokens = reloaded_tokenizers.mt.tokenize(['Illum temp ikrah madankollu.'])
tokens.numpy()

round_trip = reloaded_tokenizers.mt.detokenize(tokens)

print(round_trip.numpy()[0].decode('utf-8'))
# -



