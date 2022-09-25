# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# !pip install ipywidgets
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install sentence_splitter')
get_ipython().system('pip install -q tfds-nightly')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install jupyter-resource-usage')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install tensorflow_text')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install tensorflow-model-optimization')
get_ipython().system('pip install -U "tensorflow-text==2.8.*"')
get_ipython().system('pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116')

# +
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
import gc
import torch
import tempfile


import tempfile






gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


print(torch.cuda.memory_allocated())



# print(torch.cuda.memory_allocated())
# mt_en = tf.saved_model.load('/netscratch/abela/transformerbaseline/test_dataset/translator_exp_mt_en')
# mt_en(tf.constant("I read about triceratops in the book."))

print(torch.cuda.memory_allocated())


# +
import tensorflow_model_optimization as tfmot

model_name = '/netscratch/abela/transformerbaseline/ic_en_dataset/ted_hrlr_translate_ic_en_converter'
tokenizers = tf.saved_model.load(model_name)
tokenizers.en.get_vocab_size().numpy()

SRC_LANG = "ic"
TGT_LANG = "en"

print("DOING " + SRC_LANG + " -> " + TGT_LANG)


# Populate with typical keras callbacks

BATCH_SIZE = 64
# Compute end step to finish pruning after X epochs.
EPOCHS = 1

import tensorflow_text as text



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d // 2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))


# plt.pcolormesh(pos_encoding, cmap='RdBu')
# plt.ylabel('Depth')
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()


# Masking

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]




def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)
    
    


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding.numpy(),
        })
        return config
    
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'd_model': self.d_model,
        })
        return config
    
def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs =  tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
                units=units,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                name="encoder_layer_{}".format(i),
            )([outputs, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
    attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

def transformer(input_vocab_size,
                target_vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask =  tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
      vocab_size=input_vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
      vocab_size=target_vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=target_vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)




temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.4

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# temp_learning_rate_schedule = CustomSchedule(D_MODEL)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



transformer_test_ic_en = transformer(
    input_vocab_size=tokenizers.ic.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

transformer_test_en_ic = transformer(
    input_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.ic.get_vocab_size().numpy(),
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

    
transformer_test_ic_en.load_weights('/netscratch/abela/transformerbaseline/ic_en_dataset/translator_for_confidence_ic_en')
transformer_test_en_ic.load_weights('/netscratch/abela/transformerbaseline/ic_en_dataset/translator_for_confidence_en_ic')

# +
MAX_TOKEN_LENGTH = 80
def filter_max_tokens(en, ic):
    en = tokenizers.en.tokenize(en) 
    ic = tokenizers.ic.tokenize(ic)
    en = en.to_tensor()
    ic = ic.to_tensor()
#     
#     print(mt)
    num_tokens = tf.maximum(tf.shape(en)[1],tf.shape(ic)[1])
    return num_tokens < MAX_TOKEN_LENGTH


def tokenize_pairs(en, ic, tgt):

    en = tokenizers.en.tokenize(en) 
    ic = tokenizers.ic.tokenize(ic)
    en = en.to_tensor()
    ic = ic.to_tensor() 

    en = tf.pad(
      en,  [[0, 0,], [0,MAX_TOKEN_LENGTH-tf.shape(en)[1]]], "CONSTANT")
#     print(mt)
#     print("\n\n\n\n")
    ic = tf.pad(
      ic,  [[0, 0,], [0, MAX_TOKEN_LENGTH-tf.shape(ic)[1]]], "CONSTANT")
#     print(mt)
    if tgt == "ic":
        return (en, ic[:, :-1]), ic[:, 1:]
    else:
        return (ic, en[:, :-1]), en[:, 1:]
    
    
class TranslatorForConfidence(tf.Module):
    def __init__(self, tokenizers, transformer, tgt):
        self.tokenizers = tokenizers
        self.transformer = transformer
        self.tgt = tgt
    
    #@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string), tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence, sentence_tgt):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
            sentence_tgt = sentence_tgt[tf.newaxis]


       
        if self.tgt == "ic":
            if not filter_max_tokens(tf.constant(sentence), tf.constant(sentence_tgt)):
                return [-9999]
            (sentence,__), _ = tokenize_pairs(sentence, [''], self.tgt)
        else:
            if not filter_max_tokens(tf.constant(sentence_tgt), tf.constant(sentence)):
                return [-9999]
            (sentence,__), _ = tokenize_pairs([''],sentence,self.tgt)
            
        if self.tgt == "ic":
            # as the target is maltese, the first token to the transformer should be the
            # maltese start token.
            start_end = self.tokenizers.ic.tokenize(sentence_tgt)[0]
            start = start_end[0][tf.newaxis]
            end = start_end[-1][tf.newaxis]
        else:
            # as the target is english, the first token to the transformer should be the
            # english start token.
            start_end = self.tokenizers.en.tokenize(sentence_tgt)[0]
            start = start_end[0][tf.newaxis]
            end = start_end[-1][tf.newaxis]
      


        batch_inputs = []
            
        output_array = tf.TensorArray(dtype=tf.int64, size=MAX_TOKEN_LENGTH, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in range(MAX_TOKEN_LENGTH):
            output = tf.transpose(output_array.stack())
            batch_inputs.append([sentence, output])


           
            if [start_end[i+1]] == end:
                break
            else:
                 # concatentate the predicted_id to the output which is given to the decoder
                # as its input.
                output_array = output_array.write(i+1, [start_end[i+1]])
            
        ds = tf.data.Dataset.from_tensor_slices((batch_inputs)).map(lambda x: {'inputs': x[0], "dec_inputs": x[1]})
        total_predictions = self.transformer.predict(ds)      
        
        scores = []
        for i in range(len(total_predictions)):
#             continue
            if i == 0:
                scores.append(0)
            else:
                scores.append(tf.gather(tf.math.log(tf.nn.softmax(total_predictions[i][i-1])), start_end[i-1]))     
        return scores
model_name = '/netscratch/abela/transformerbaseline/ic_en_dataset/ted_hrlr_translate_ic_en_converter'
tokenizers = tf.saved_model.load(model_name)
ground_truth = "my problem"
sentence = "problema tieghi"
translator_for_confidence_ic_en = TranslatorForConfidence(tokenizers, transformer_test_ic_en, "ic") 
translator_for_confidence_en_ic = TranslatorForConfidence(tokenizers, transformer_test_en_ic, "en")    
# -

# %%time
res = translator_for_confidence_ic_en(tf.constant(sentence), tf.constant(ground_truth))
print(res)

# !nvidia-smi

# +
import math
import langid
from transformers import AutoTokenizer, AutoModelForMaskedLM

import numpy as np

import tensorflow_text
import math
from joblib import Parallel, delayed
tokenizer_ic = AutoTokenizer.from_pretrained("neurocode/IsRoBERTa", use_fast=True)
model_ic = AutoModelForMaskedLM.from_pretrained("neurocode/IsRoBERTa")
# set device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# put model on GPU (is done in-place)
model_ic.to(device)
model_ic.eval()




# +

tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model_en = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# set device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"
print(device)
# put model on GPU (is done in-place)
model_en.to(device)
model_en.eval()

softmax = torch.nn.Softmax(dim=0)

# +


print(device)
def lm_en_prob(sentence):

    tokenize_input = tokenizer_en(sentence, truncation=True, max_length=512, padding=True, return_tensors='pt')["input_ids"].cuda()

    
    tokenize_input.to(device)
    loss = model_en(tokenize_input.cuda())[0]
    total_scores = []
    
    for j, individual_score in enumerate(loss):
        
        scores = []
        
        length_of_input = len(tokenize_input[j])
        softmax_of_sent = softmax(loss[j])
        
        logs_of_softmax_of_sent = torch.log(softmax_of_sent)
        for i in range(length_of_input):
            scores.append(logs_of_softmax_of_sent[i][tokenize_input[j, i]])
        total_scores.append(scores)
    del tokenize_input, loss, softmax_of_sent, logs_of_softmax_of_sent, scores
    return total_scores


def lm_ic_prob(sentence):
    tokenize_input = tokenizer_en(sentence, truncation=True, max_length=512, padding=True, return_tensors='pt')["input_ids"].cuda()

    tokenize_input.to(device)
    loss = model_ic(tokenize_input.cuda())[0]
    total_scores = []
    
    for j, individual_score in enumerate(loss):
        scores = []
        length_of_input = len(tokenize_input[j])
        softmax_of_sent = softmax(loss[j])
        
        logs_of_softmax_of_sent = torch.log(softmax_of_sent)
        for i in range(length_of_input):
            scores.append(logs_of_softmax_of_sent[i][tokenize_input[j, i]])
        total_scores.append(scores)
        
    del tokenize_input, loss, softmax_of_sent, logs_of_softmax_of_sent, scores
    return total_scores


def mt_en_prob(sentence_en, sentence_ic):
#     return [0]
    return translator_for_confidence_en_ic(tf.constant(sentence_en), tf.constant(sentence_ic))



def mt_ic_prob(sentence_ic, sentence_en):
#     return [0]
    return translator_for_confidence_ic_en(tf.constant(sentence_ic), tf.constant(sentence_en))



def sentence_score(sentence_en, sentence_ic, score):
    lm_en = lm_en_prob(sentence_en)
    lm_ic = lm_ic_prob(sentence_ic)

    for i, sentence in enumerate(sentence_en):
        if score[i] == 0:
            mt_en = mt_en_prob(sentence, sentence_ic[i])
            mt_ic = mt_ic_prob(sentence_ic[i], sentence)
            lm_scores = ((1 / len(lm_en[i])) * sum(lm_en[i])) + (
                    (1 / len(lm_ic[i])) * sum(lm_ic[i]))
            mt_scores = ((1 / len(mt_en)) * sum(mt_en)) + (
                    (1 / len(mt_ic)) * sum(mt_ic))

            # print(lm_scores)
            score[i] = (1 / 4) * (lm_scores.cpu() + mt_scores)
    try:
        del lm_en, lm_ic, mt_en, mt_ic, lm_scores, mt_scores
    except:
        print("Can't delete...")
    return score
print(torch.cuda.memory_allocated())
# -

batch_size = 20

# +
print(torch.cuda.memory_allocated())

if __name__ == '__main__':
    for path in ["train", "dev", "test"]:
    #for path in ["dev"]:
        lines_to_remove = []
        with open('../original_data/en-ic/' + path + '.en', "r", encoding="utf-8") as f:
            print("read1")
            sentences_en = f.read().split('\n')
            with open('../original_data/en-ic/' + path + '.ic', "r", encoding="utf-8") as f:
                print("read2")
                sentences_ic = f.read().split('\n')
                length_sentences_en = len(sentences_en)
                # result = Parallel()(delayed(inner_for)(i,sentences_en, sentences_mt, lines_to_remove) for i in range(30 - 1))
                
                for i in range(math.ceil((length_sentences_en - 1) / batch_size)):
                    print("\n")
                    print(i)
                    torch.cuda.empty_cache() # PyTorch thing
                    with torch.no_grad():
                        print(i*batch_size)
                        if (i*batch_size)+batch_size >= length_sentences_en:
                            sentence_en = sentences_en[(i*batch_size):length_sentences_en]
                            sentence_ic = sentences_ic[(i*batch_size):length_sentences_en]
                            scores = np.zeros(length_sentences_en-i*batch_size)
                        else:
                            sentence_en = sentences_en[i*batch_size:(i*batch_size)+batch_size]
                            sentence_ic = sentences_ic[i*batch_size:(i*batch_size)+batch_size]
                            scores = np.zeros(batch_size)

                        for j, sentence in enumerate(sentence_en):
                            response = langid.classify(sentence)
                            if response[0] != "en":
                                # print("Not English: " + sentence_en)
                                scores[j] = -1

                        for j, sentence in enumerate(sentence_ic):
                            response = langid.classify(sentence)
                            if response[0] != "is":
                                # print("Not Maltese: " + sentence_mt)
                                scores[j] = -1


                        scores = sentence_score(sentence_en, sentence_ic, scores)
#                         print(scores)
                        for score in scores:
                            if score < -7:
                                lines_to_remove.append(i)
    #                 result = Parallel()(delayed(inner_for)(i,sentences_en, sentences_mt, lines_to_remove, length_sentences_en) for i in range(math.ceil((length_sentences_en - 1) / batch_size)))
                
                print("LINES TO REMOVE: ")
                print(lines_to_remove)

        with open('../original_data/en-ic/' + path + '.en', "w", encoding="utf-8") as en, open('../original_data/en-ic/' + path + '.ic', "w", encoding="utf-8") as ic:
            for pos, line in enumerate(sentences_en):
                if pos not in lines_to_remove:
                    en.write(line+"\n")
                    ic.write(sentences_ic[pos]+"\n")
                else:
                    print(line)
                    print(sentences_ic[pos])

# -


