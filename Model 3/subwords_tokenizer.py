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


# os.system("tfds build")


"""## Download the dataset

Fetch the Maltese/English translation dataset from [tfds](https://tensorflow.org/datasets):
"""

examples, metadata = tfds.load(name="test_dataset", with_info=True,
                               as_supervised=True, data_dir="test_dataset")
train_examples, val_examples = examples['train'], examples['validation']

"""This dataset produces Maltese/English sentence pairs:"""

for en, mt in train_examples.take(1):
    print("Maltese: ", mt.numpy().decode('utf-8'))
    print("English:   ", en.numpy().decode('utf-8'))

"""Note a few things about the example sentences above:
* They're lower case.
* There are spaces around the punctuation.
* It's not clear if or what unicode normalization is being used.
"""

train_en = train_examples.map(lambda en, mt: en)
train_mt = train_examples.map(lambda en, mt: mt)



from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab



bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=32000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)
#

# +

# # Commented out IPython magic to ensure Python compatibility.
# # # %%time
mt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_mt.batch(1000).prefetch(2),
    **bert_vocab_args
)
#
# """Here are some slices of the resulting vocabulary."""

print(mt_vocab[:10])
print(mt_vocab[100:110])
print(mt_vocab[1000:1010])
print(mt_vocab[-10:])

"""Write a vocabulary file:"""


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file('mt_vocab.txt', mt_vocab)

# """Use that function to generate a vocabulary from the english data:"""
#
# # Commented out IPython magic to ensure Python compatibility.
# # # %%time
en_vocab = bert_vocab.bert_vocab_from_dataset(
    train_en.batch(1000).prefetch(2),
    **bert_vocab_args
)

print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])

"""Here are the two vocabulary files:"""

write_vocab_file('en_vocab.txt', en_vocab)

"""## Build the tokenizer
<a id="build_the_tokenizer"></a>

The `text.BertTokenizer` can be initialized by passing the vocabulary file's path as the first argument (see the section on [tf.lookup](#tf.lookup) for other options):
"""



# +
bert_tokenizer_params = dict(lower_case=True)
mt_tokenizer = text.BertTokenizer('mt_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)

"""Now you can use it to encode some text. Take a batch of 3 examples from the english data:"""

for mt_examples, en_examples in train_examples.batch(3).take(1):
    for ex in en_examples:
        print(ex.numpy())

"""Run it through the `BertTokenizer.tokenize` method. Initially, this returns a `tf.RaggedTensor` with axes `(batch, word, word-piece)`:"""

# Tokenize the examples -> (batch, word, word-piece)
token_batch = en_tokenizer.tokenize(en_examples)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2, -1)

for ex in token_batch.to_list():
    print(ex)

"""If you replace the token IDs with their text representations (using `tf.gather`) you can see that in the first example the words `"searchability"` and  `"serendipity"` have been decomposed into `"search ##ability"` and `"s ##ere ##nd ##ip ##ity"`:"""



"""To re-assemble words from the extracted tokens, use the `BertTokenizer.detokenize` method:"""

words = en_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)

"""> Note: `BertTokenizer.tokenize`/`BertTokenizer.detokenize` does not round
trip losslessly. The result of `detokenize` will not, in general, have the
same content or offsets as the input to `tokenize`. This is because of the
"basic tokenization" step, that splits the strings into words before
applying the `WordpieceTokenizer`, includes irreversible
steps like lower-casing and splitting on punctuation. `WordpieceTokenizer`
on the other hand **is** reversible.

## Customization and export

This tutorial builds the text tokenizer and detokenizer used by the [Transformer](https://tensorflow.org/text/tutorials/transformer) tutorial. This section adds methods and processing steps to simplify that tutorial, and exports the tokenizers using `tf.saved_model` so they can be imported by the other tutorials.

### Custom tokenization

The downstream tutorials both expect the tokenized text to include `[START]` and `[END]` tokens.

The `reserved_tokens` reserve space at the beginning of the vocabulary, so `[START]` and `[END]` have the same indexes for both languages:
"""
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], START)
    ends = tf.fill([count,1], END)
    return tf.concat([starts, ragged, ends], axis=1)

words = en_tokenizer.detokenize(add_start_end(token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)

"""### Custom detokenization

Before exporting the tokenizers there are a couple of things you can cleanup for the downstream tutorials:

1. They want to generate clean text output, so drop reserved tokens like `[START]`, `[END]` and `[PAD]`.
2. They're interested in complete strings, so apply a string join along the `words` axis of the result.
"""


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

token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2, -1)
words = en_tokenizer.detokenize(token_batch)
print(words)

cleanup_text(reserved_tokens, words).numpy()

"""### Export

The following code block builds a `CustomTokenizer` class to contain the `text.BertTokenizer` instances, the custom logic, and the `@tf.function` wrappers required for export.
"""


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


"""Build a `CustomTokenizer` for each language:"""

tokenizers = tf.Module()
tokenizers.mt = CustomTokenizer(reserved_tokens, 'mt_vocab.txt')
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')

"""Export the tokenizers as a `saved_model`:"""

model_name = 'ted_hrlr_translate_mt_en_converter'
tf.saved_model.save(tokenizers, model_name)

"""Reload the `saved_model` and test the methods:"""

reloaded_tokenizers = tf.saved_model.load(model_name)
reloaded_tokenizers.en.get_vocab_size().numpy()

tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
tokens.numpy()

text_tokens = reloaded_tokenizers.en.detokenize(tokens)
text_tokens

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
# -


