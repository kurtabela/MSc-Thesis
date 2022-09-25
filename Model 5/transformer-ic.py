# %%
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install sentence_splitter')
get_ipython().system('pip install -q tfds-nightly')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install jupyter-resource-usage')
get_ipython().system('pip install tensor2tensor')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install tensorflow-model-optimization')
get_ipython().system('pip install tokenizer')
get_ipython().system('pip install tensorflow-text --force-reinstall')

# %%
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

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow import keras
from tensor2tensor.layers import common_layers
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest
import tensorflow.compat.v1 as tf1
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
# tf.debugging.set_log_device_placement(True)
import dataset_without_masks_ic

import tempfile
os.chdir("dataset_without_masks_ic")
#os.system("tfds build")


os.system("tfds build  --data_dir " + os.getcwd())


# %%
from tensor2tensor.layers import common_layers


# %%
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest

# %%



examples, metadata = tfds.load(name="dataset_without_masks_ic", with_info=True,
                               as_supervised=True, data_dir=os.getcwd())
train_examples, val_examples= examples['train'], examples['validation']



#train_examples, val_examples= examples['train'], examples['validation']



# %%
print(examples)
print(metadata)

# %%
for en, ic in train_examples.take(2):
    print(en)
    print(ic)

# %%
model_name = 'ted_hrlr_translate_ic_en_converter'
tokenizers = tf.saved_model.load(model_name)
tokenizers.en.get_vocab_size().numpy()
MAX_TOKEN_LENGTH = 80
#
## %%
#lengths = []
#
#for en_examples, ic_examples in train_examples.batch(1024):
#    ic_tokens = tokenizers.ic.tokenize(ic_examples)
#    lengths.append(ic_tokens.row_lengths())
#
#    en_tokens = tokenizers.en.tokenize(en_examples)
#    lengths.append(en_tokens.row_lengths())
#    print('.', end='', flush=True)
#
## %%
#all_lengths = np.concatenate(lengths)
#
#plt.hist(all_lengths, np.linspace(0, 500, 101))
#plt.ylim(plt.ylim())
#max_length = max(all_lengths)
#plt.plot([max_length, max_length], plt.ylim())
#plt.title(f'Max tokens per example: {max_length}');
#
## %%
##Keep at 80
#
#
#def condition(x):
#    return x < MAX_TOKEN_LENGTH
#
#print("Average tokens per example: ")
#print(sum(all_lengths) / len(all_lengths))
#
#print("Number of elements smaller than len " + str(MAX_TOKEN_LENGTH))
#satisfyCond = sum(condition(x) for x in all_lengths)
#print(satisfyCond)
#
#totalLengths = len(all_lengths)
#print("Total lengths: " + str(totalLengths))
#
#print("Coverage: (%)")
#print(satisfyCond/totalLengths)

# %%
SRC_LANG = "ic"
TGT_LANG = "en"

print("DOING " + SRC_LANG + " -> " + TGT_LANG)

# %%
tokens = tokenizers.ic.tokenize(['Hawn illum!'])
tokens.numpy()

# %%
text_tokens = tokenizers.ic.detokenize(tokens)
text_tokens

# %%
import tensorflow_model_optimization as tfmot

# Populate with typical keras callbacks

BATCH_SIZE = 50
# Compute end step to finish pruning after X epochs.
EPOCHS = 10

train_samples = train_examples.cardinality().numpy()
print(train_samples)

num_samples = val_examples.cardinality().numpy()
print(num_samples)

# %%

import tensorflow_text as text
MAX_TOKEN_LENGTH = 80

def filter_max_tokens(en, ic):
    en = tokenizers.en.tokenize([en]) 
    ic = tokenizers.ic.tokenize([ic])
    en = en.to_tensor()
    ic = ic.to_tensor()
#     
#     print(ic)
#     print(tf.shape(ic))
    num_tokens = tf.maximum(tf.shape(en)[1],tf.shape(ic)[1])
    return num_tokens < MAX_TOKEN_LENGTH

# filter_max_tokens(tf.constant(["Test"]), tf.constant(["test jekk tara gejja"]))


def tokenize_pairs(en, ic):

    en = tokenizers.en.tokenize(en) 
    ic = tokenizers.ic.tokenize(ic)
    en = en.to_tensor()
    ic = ic.to_tensor() 

    en = tf.pad(
      en,  [[0, 0,], [0,MAX_TOKEN_LENGTH-tf.shape(en)[1]]], "CONSTANT")
#     print(ic)
#     print("\n\n\n\n")
    ic = tf.pad(
      ic,  [[0, 0,], [0, MAX_TOKEN_LENGTH-tf.shape(ic)[1]]], "CONSTANT")
#     print(ic)
    if SRC_LANG == "en" and TGT_LANG == "ic":
        return (en, ic[:, :-1]), ic[:, 1:]
    else:
        return (ic, en[:, :-1]), en[:, 1:]
    


BUFFER_SIZE = 20000


def make_batches(ds):
    
    
#     tmp = ds.batch(1).filter(filter_max_tokens).map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        
#     dataset = tmp.cache().batch(BATCH_SIZE)
#     return(dataset)


    return (
            ds
              .cache()
#              .shuffle(BUFFER_SIZE)
              .filter(filter_max_tokens)
              .batch(BATCH_SIZE)
              .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

train_batches = make_batches(train_examples)
# train_samples = len(list(train_batches))
# print(train_samples)
val_batches = make_batches(val_examples)

#num_of_train_batches = len(list(train_batches))


# %%
#print(num_of_train_batches)


# %%

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

# %%


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])




# %%
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
#@tf.keras.utils.register_keras_serializable    
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


# %%

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
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    if SRC_LANG == "en" and TGT_LANG == "ic":
        transformer_test = transformer(
            input_vocab_size=tokenizers.en.get_vocab_size().numpy(),
            target_vocab_size=tokenizers.ic.get_vocab_size().numpy(),
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

    else:
        transformer_test = transformer(
            input_vocab_size=tokenizers.ic.get_vocab_size().numpy(),
            target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def accuracy_function(real, pred):
        accuracies = tf.equal(tf.cast(real, dtype=tf.int64), tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')



# %%
# Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every `n` epochs.


checkpoint_path = "./checkpoints/train/" +  SRC_LANG + "_" + TGT_LANG + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
print(checkpoint_path)
print(checkpoint_dir)
if latest:
    print("LATEST IS LOADED")
    transformer_test.load_weights(checkpoint_path)
    print(checkpoint_path)

# %%
transformer_test.summary()

# %%

transformer_test.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy_function])




# %%
transformer_test.fit(train_batches, epochs=EPOCHS, validation_data=val_batches, callbacks = [cp_callback])

# %%

# Assuming EOS_ID is 3
EOS_ID = 3
# Default value for INF
INF = 1. * 1e7


def _merge_beam_dim(tensor):
  """Reshapes first two dimensions in to single dimension.
  Args:
    tensor: Tensor to reshape of shape [A, B, ...]
  Returns:
    Reshaped tensor of shape [A*B, ...]
  """
  shape = common_layers.shape_list(tensor)
  shape[0] *= shape[1]  # batch -> batch * beam_size
  shape.pop(1)  # Remove beam dim
  return tf1.reshape(tensor, shape)


def _unmerge_beam_dim(tensor, batch_size, beam_size):
  """Reshapes first dimension back to [batch_size, beam_size].
  Args:
    tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
    batch_size: Tensor, original batch size.
    beam_size: int, original beam size.
  Returns:
    Reshaped tensor of shape [batch_size, beam_size, ...]
  """
  shape = common_layers.shape_list(tensor)
  new_shape = [batch_size] + [beam_size] + shape[1:]
  return tf1.reshape(tensor, new_shape)


def _expand_to_beam_size(tensor, beam_size):
  """Tiles a given tensor by beam_size.
  Args:
    tensor: tensor to tile [batch_size, ...]
    beam_size: How much to tile the tensor by.
  Returns:
    Tiled tensor [batch_size, beam_size, ...]
  """
  tensor = tf1.expand_dims(tensor, axis=1)
  tile_dims = [1] * tensor.shape.ndims
  tile_dims[1] = beam_size

  return tf1.tile(tensor, tile_dims)


def get_state_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  shape = tensor.shape.as_list()
  for i in range(1, len(shape) - 1):
    shape[i] = None
  return tf1.TensorShape(shape)


def compute_batch_indices(batch_size, beam_size):
  """Computes the i'th coordinate that contains the batch index for gathers.
  Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
  batch the beam item is in. This will create the i of the i,j coordinate
  needed for the gather.
  Args:
    batch_size: Batch size
    beam_size: Size of the beam.
  Returns:
    batch_pos: [batch_size, beam_size] tensor of ids
  """
  batch_pos = tf1.range(batch_size * beam_size) // beam_size
  batch_pos = tf1.reshape(batch_pos, [batch_size, beam_size])
  return batch_pos


def fast_tpu_gather(params, indices, name=None):
  """Fast gather implementation for models running on TPU.
  This function use one_hot and batch matmul to do gather, which is faster
  than gather_nd on TPU. For params that have dtype of int32 (sequences to
  gather from), batch_gather is used to keep accuracy.
  Args:
    params: A tensor from which to gather values.
      [batch_size, original_size, ...]
    indices: A tensor used as the index to gather values.
      [batch_size, selected_size].
    name: A string, name of the operation (optional).
  Returns:
    gather_result: A tensor that has the same rank as params.
      [batch_size, selected_size, ...]
  """
  with tf1.name_scope(name):
    dtype = params.dtype

    def _gather(params, indices):
      """Fast gather using one_hot and batch matmul."""
      if dtype != tf1.float32:
        params = tf1.cast(params, tf1.float32)
      shape = common_layers.shape_list(params)
      indices_shape = common_layers.shape_list(indices)
      ndims = params.shape.ndims
      # Adjust the shape of params to match one-hot indices, which is the
      # requirement of Batch MatMul.
      if ndims == 2:
        params = tf1.expand_dims(params, axis=-1)
      if ndims > 3:
        params = tf1.reshape(params, [shape[0], shape[1], -1])
      gather_result = tf1.matmul(
          tf1.one_hot(indices, shape[1], dtype=params.dtype), params)
      if ndims == 2:
        gather_result = tf1.squeeze(gather_result, axis=-1)
      if ndims > 3:
        shape[1] = indices_shape[1]
        gather_result = tf1.reshape(gather_result, shape)
      if dtype != tf1.float32:
        gather_result = tf1.cast(gather_result, dtype)
      return gather_result

    # If the dtype is int, use the gather instead of one_hot matmul to avoid
    # precision loss. The max int value can be represented by bfloat16 in MXU is
    # 256, which is smaller than the possible id values. Encoding/decoding can
    # potentially used to make it work, but the benenfit is small right now.
    if dtype.is_integer:
      gather_result = tf1.batch_gather(params, indices)
    else:
      gather_result = _gather(params, indices)

    return gather_result


def _create_make_unique(inputs):
  """Replaces the lower bits of each element with iota.
  The iota is used to derive the index, and also serves the purpose to
  make each element unique to break ties.
  Args:
    inputs: A tensor with rank of 2 and dtype of tf1.float32.
      [batch_size, original_size].
  Returns:
    A tensor after element wise transformation, with dtype the same as inputs.
    [batch_size, original_size].
  Raises:
    ValueError: If the rank of the input tensor does not equal 2.
  """
  if inputs.shape.ndims != 2:
    raise ValueError("Input of top_k_with_unique must be rank-2 "
                     "but got: %s" % inputs.shape)

  height = inputs.shape[0]
  width = inputs.shape[1]
  zeros = tf1.zeros([height, width], dtype=tf1.int64)

  # Count_mask is used to mask away the low order bits to ensure that every
  # element is distinct.
  log2_ceiling = int(math.ceil(math.log(int(width), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = ~(next_power_of_two - 1)
  count_mask_r0 = tf1.constant(count_mask)
  count_mask_r2 = tf1.fill([height, width], count_mask_r0)

  # Smallest_normal is the bit representation of the smallest positive normal
  # floating point number. The sign is zero, exponent is one, and the fraction
  # is zero.
  smallest_normal = 1 << 23
  smallest_normal_r0 = tf1.constant(smallest_normal, dtype=tf1.int64)
  smallest_normal_r2 = tf1.fill([height, width], smallest_normal_r0)

  # Low_bit_mask is used to mask away the sign bit when computing the absolute
  # value.
  low_bit_mask = ~(1 << 31)
  low_bit_mask_r0 = tf1.constant(low_bit_mask, dtype=tf1.int64)
  low_bit_mask_r2 = tf1.fill([height, width], low_bit_mask_r0)

  iota = tf1.tile(tf1.expand_dims(tf1.range(width, dtype=tf1.int64), 0),
                 [height, 1])

  # Compare the absolute value with positive zero to handle negative zero.
  input_r2 = tf1.bitcast(inputs, tf1.int64)
  abs_r2 = tf1.bitwise.bitwise_and(input_r2, low_bit_mask_r2)
  if_zero_r2 = tf1.equal(abs_r2, zeros)
  smallest_normal_preserving_sign_r2 = tf1.bitwise.bitwise_or(
      input_r2, smallest_normal_r2)
  input_no_zeros_r2 = tf1.where(
      if_zero_r2, smallest_normal_preserving_sign_r2, input_r2)

  # Discard the low-order bits and replace with iota.
  and_r2 = tf1.bitwise.bitwise_and(input_no_zeros_r2, count_mask_r2)
  or_r2 = tf1.bitwise.bitwise_or(and_r2, iota)
  return tf1.bitcast(or_r2, tf1.float32)


def _create_topk_unique(inputs, k):
  """Creates the top k values in sorted order with indices.
  Args:
    inputs: A tensor with rank of 2. [batch_size, original_size].
    k: An integer, number of top elements to select.
  Returns:
    topk_r2: A tensor, the k largest elements. [batch_size, k].
    topk_indices_r2: A tensor, indices of the top k values. [batch_size, k].
  """
  height = inputs.shape[0]
  width = inputs.shape[1]
  neg_inf_r0 = tf1.constant(-np.inf, dtype=tf1.float32)
  ones = tf1.ones([height, width], dtype=tf1.float32)
  neg_inf_r2 = ones * neg_inf_r0
  inputs = tf1.where(tf1.is_nan(inputs), neg_inf_r2, inputs)

  # Select the current largest value k times and keep them in topk_r2. The
  # selected largest values are marked as the smallest value to avoid being
  # selected again.
  tmp = inputs
  topk_r2 = tf1.zeros([height, k], dtype=tf1.float32)
  for i in range(k):
    kth_order_statistic = tf1.reduce_max(tmp, axis=1, keepdims=True)
    k_mask = tf1.tile(tf1.expand_dims(tf1.equal(tf1.range(k), tf1.fill([k], i)), 0),
                     [height, 1])
    topk_r2 = tf1.where(k_mask, tf1.tile(kth_order_statistic, [1, k]), topk_r2)
    ge_r2 = tf1.greater_equal(inputs, tf1.tile(kth_order_statistic, [1, width]))
    tmp = tf1.where(ge_r2, neg_inf_r2, inputs)

  log2_ceiling = int(math.ceil(math.log(float(int(width)), 2)))
  next_power_of_two = 1 << log2_ceiling
  count_mask = next_power_of_two - 1
  mask_r0 = tf1.constant(count_mask)
  mask_r2 = tf1.fill([height, k], mask_r0)
  topk_r2_s32 = tf1.bitcast(topk_r2, tf1.int64)
  topk_indices_r2 = tf1.bitwise.bitwise_and(topk_r2_s32, mask_r2)
  return topk_r2, topk_indices_r2


def top_k_with_unique(inputs, k):
  """Finds the values and indices of the k largests entries.
  Instead of doing sort like tf1.nn.top_k, this function finds the max value
  k times. The running time is proportional to k, which is be faster when k
  is small. The current implementation supports only inputs of rank 2.
  In addition, iota is used to replace the lower bits of each element, this
  makes the selection more stable when there are equal elements. The
  overhead is that output values are approximated.
  Args:
    inputs: A tensor with rank of 2. [batch_size, original_size].
    k: An integer, number of top elements to select.
  Returns:
    top_values: A tensor, the k largest elements in sorted order.
      [batch_size, k].
    indices: A tensor, indices of the top_values. [batch_size, k].
  """
  unique_inputs = _create_make_unique(tf1.cast(inputs, tf1.float32))
  top_values, indices = _create_topk_unique(unique_inputs, k)
  top_values = tf1.cast(top_values, inputs.dtype)
  return top_values, indices


def compute_topk_scores_and_seq(sequences,
                                scores,
                                scores_to_gather,
                                flags,
                                beam_size,
                                batch_size,
                                prefix="default",
                                states_to_gather=None,
                                use_tpu=False,
                                use_top_k_with_unique=True):
  """Given sequences and scores, will gather the top k=beam size sequences.
  This function is used to grow alive, and finished. It takes sequences,
  scores, and flags, and returns the top k from sequences, scores_to_gather,
  and flags based on the values in scores.
  This method permits easy introspection using tf1dbg.  It adds three named ops
  that are prefixed by `prefix`:
    - _topk_seq: the tensor for topk_seq returned by this method.
    - _topk_flags: the tensor for topk_finished_flags returned by this method.
    - _topk_scores: the tensor for tokp_gathered_scores returned by this method.
  Args:
    sequences: Tensor of sequences that we need to gather from.
      [batch_size, beam_size, seq_length]
    scores: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will use these to compute the topk.
    scores_to_gather: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will return the gathered scores from here.
      Scores to gather is different from scores because for grow_alive, we will
      need to return log_probs, while for grow_finished, we will need to return
      the length penalized scores.
    flags: Tensor of bools for sequences that say whether a sequence has reached
      EOS or not
    beam_size: int
    batch_size: int
    prefix: string that will prefix unique names for the ops run.
    states_to_gather: dict (possibly nested) of decoding states.
    use_tpu: A bool, whether to compute topk scores and sequences on TPU.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during TPU beam search.
  Returns:
    Tuple of
    (topk_seq [batch_size, beam_size, decode_length],
     topk_gathered_scores [batch_size, beam_size],
     topk_finished_flags[batch_size, beam_size])
  """
  if not use_tpu:
    _, topk_indexes = tf1.nn.top_k(scores, k=beam_size)
    # The next three steps are to create coordinates for tf1.gather_nd to pull
    # out the topk sequences from sequences based on scores.
    # batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    # batch the beam item is in. This will create the i of the i,j coordinate
    # needed for the gather
    batch_pos = compute_batch_indices(batch_size, beam_size)

    # top coordinates will give us the actual coordinates to do the gather.
    # stacking will create a tensor of dimension batch * beam * 2, where the
    # last dimension contains the i,j gathering coordinates.
    top_coordinates = tf1.stack([batch_pos, topk_indexes], axis=2)

    # Gather up the highest scoring sequences.  For each operation added, give
    # it a concrete name to simplify observing these operations with tf1dbg.
    # Clients can capture these tensors by watching these node names.
    def gather(tensor, name):
      return tf1.gather_nd(tensor, top_coordinates, name=(prefix + name))
    topk_seq = gather(sequences, "_topk_seq")
    topk_flags = gather(flags, "_topk_flags")
    topk_gathered_scores = gather(scores_to_gather, "_topk_scores")
    if states_to_gather:
      topk_gathered_states = nest.map_structure(
          lambda state: gather(state, "_topk_states"), states_to_gather)
    else:
      topk_gathered_states = states_to_gather
  else:
    if use_top_k_with_unique:
      _, topk_indexes = top_k_with_unique(scores, k=beam_size)
    else:
      _, topk_indexes = tf1.nn.top_k(scores, k=beam_size)
    # Gather up the highest scoring sequences.  For each operation added, give
    # it a concrete name to simplify observing these operations with tf1dbg.
    # Clients can capture these tensors by watching these node names.
    topk_seq = fast_tpu_gather(sequences, topk_indexes, prefix + "_topk_seq")
    topk_flags = fast_tpu_gather(flags, topk_indexes, prefix + "_topk_flags")
    topk_gathered_scores = fast_tpu_gather(scores_to_gather, topk_indexes,
                                           prefix + "_topk_scores")
    if states_to_gather:
      topk_gathered_states = nest.map_structure(
          # pylint: disable=g-long-lambda
          lambda state: fast_tpu_gather(state, topk_indexes,
                                        prefix + "_topk_states"),
          states_to_gather)
    else:
      topk_gathered_states = states_to_gather
  return topk_seq, topk_gathered_scores, topk_flags, topk_gathered_states


def beam_search(symbols_to_logits_fn,
                initial_ids,
                beam_size,
                decode_length,
                vocab_size,
                alpha,
                encoder_input,
                states=None,
                eos_id=EOS_ID,
                stop_early=True,
                use_tpu=False,
                use_top_k_with_unique=True):
  """Beam search with length penalties.
  Requires a function that can take the currently decoded symbols and return
  the logits for the next symbol. The implementation is inspired by
  https://arxiv.org/abs/1609.08144.
  When running, the beam search steps can be visualized by using tf1dbg to watch
  the operations generating the output ids for each beam step.  These operations
  have the pattern:
    (alive|finished)_topk_(seq,scores)
  Operations marked `alive` represent the new beam sequences that will be
  processed in the next step.  Operations marked `finished` represent the
  completed beam sequences, which may be padded with 0s if no beams finished.
  Operations marked `seq` store the full beam sequence for the time step.
  Operations marked `scores` store the sequence's final log scores.
  The beam search steps will be processed sequentially in order, so when
  capturing observed from these operations, tensors, clients can make
  assumptions about which step is being recorded.
  WARNING: Assumes 2nd dimension of tensors in `states` and not invariant, this
  means that the shape of the 2nd dimension of these tensors will not be
  available (i.e. set to None) inside symbols_to_logits_fn.
  Args:
    symbols_to_logits_fn: Interface to the model, to provide logits.
        Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size]
    initial_ids: Ids to start off the decoding, this will be the first thing
        handed to symbols_to_logits_fn (after expanding to beam size)
        [batch_size]
    beam_size: Size of the beam.
    decode_length: Number of steps to decode for.
    vocab_size: Size of the vocab, must equal the size of the logits returned by
        symbols_to_logits_fn
    alpha: alpha for length penalty.
    states: dict (possibly nested) of decoding states.
    eos_id: ID for end of sentence.
    stop_early: a boolean - stop once best sequence is provably determined.
    use_tpu: A bool, whether to do beam search on TPU.
    use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
      top_k during TPU beam search.
  Returns:
    Tuple of
    (decoded beams [batch_size, beam_size, decode_length]
     decoding probabilities [batch_size, beam_size])
  """

  batch_size = common_layers.shape_list(initial_ids)[0]

  # Assume initial_ids are prob 1.0
  initial_log_probs = tf1.constant([[0.] + [-INF] * (beam_size - 1)])
  # Expand to beam_size (batch_size, beam_size)
  alive_log_probs = tf1.tile(initial_log_probs, [batch_size, 1])

  # Expand each batch and state to beam_size
  alive_seq = _expand_to_beam_size(initial_ids, beam_size)
  alive_seq = tf1.expand_dims(alive_seq, axis=2)  # (batch_size, beam_size, 1)

  states = {}

  # Finished will keep track of all the sequences that have finished so far
  # Finished log probs will be negative infinity in the beginning
  # finished_flags will keep track of booleans
  finished_seq = tf1.zeros(common_layers.shape_list(alive_seq), tf1.int64)
  # Setting the scores of the initial to negative infinity.
  finished_scores = tf1.ones([batch_size, beam_size]) * -INF
  finished_flags = tf1.zeros([batch_size, beam_size], tf1.bool)

  def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
    """Given sequences and scores, will gather the top k=beam size sequences.
    Args:
      finished_seq: Current finished sequences.
        [batch_size, beam_size, current_decoded_length]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, current_decoded_length]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    if not use_tpu:
      # First append a column of 0'ids to finished to make the same length with
      # finished scores
      finished_seq = tf1.concat(
          [finished_seq,
           tf1.zeros([batch_size, beam_size, 1], tf1.int64)], axis=2)


    # Set the scores of the unfinished seq in curr_seq to large negative
    # values
    curr_scores += (1. - tf1.cast(curr_finished, tf1.float32)) * -INF
    # concatenating the sequences and scores along beam axis
    curr_finished_seq = tf1.concat([finished_seq, curr_seq], axis=1)
    curr_finished_scores = tf1.concat([finished_scores, curr_scores], axis=1)
    curr_finished_flags = tf1.concat([finished_flags, curr_finished], axis=1)
    return compute_topk_scores_and_seq(
        curr_finished_seq,
        curr_finished_scores,
        curr_finished_scores,
        curr_finished_flags,
        beam_size,
        batch_size,
        "grow_finished",
        use_tpu=False,
        use_top_k_with_unique=use_top_k_with_unique)

  def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished, states):
    """Given sequences and scores, will gather the top k=beam size sequences.
    Args:
      curr_seq: current topk sequence that has been grown by one position.
        [batch_size, beam_size, i+1]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_log_probs: log probs for each of these sequences.
        [batch_size, beam_size]
      curr_finished: Finished flags for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    # Set the scores of the finished seq in curr_seq to large negative
    # values
    curr_scores += tf1.cast(curr_finished, tf1.float32) * -INF
    return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                       curr_finished, beam_size, batch_size,
                                       "grow_alive", states, use_tpu=False)

  def grow_topk(i, alive_seq, alive_log_probs, states):
    r"""Inner beam search loop.
    This function takes the current alive sequences, and grows them to topk
    sequences where k = 2*beam. We use 2*beam because, we could have beam_size
    number of sequences that might hit <EOS> and there will be no alive
    sequences to continue. With 2*beam_size, this will not happen. This relies
    on the assumption the vocab size is > beam size. If this is true, we'll
    have at least beam_size non <EOS> extensions if we extract the next top
    2*beam words.
    Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
    https://arxiv.org/abs/1609.08144.
    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
      alive_log_probs: probabilities of these sequences. [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Topk sequences extended by the next word,
         The log probs of these sequences,
         The scores with length penalty of these sequences,
         Flags indicating which of these sequences have finished decoding,
         dict of transformed decoding states)
    """
    

    # Get the logits for all the possible next symbols

    flat_ids = tf1.reshape(alive_seq, [batch_size * beam_size, -1])
    encoder_input_reshaped = tf1.repeat(encoder_input, repeats=[beam_size], axis=0)	
    # (batch_size * beam_size, decoded_length)
    if states:
      flat_states = nest.map_structure(_merge_beam_dim, states)
      flat_logits, flat_states = symbols_to_logits_fn(flat_ids, i, flat_states)
      states = nest.map_structure(
          lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_states)
    else:
      batch_inputs = []
      padding_applied = []	      
#       print("num of flat ids: ")	
#       print(len(flat_ids))	
#       print("num of enc inputs: ")	
#       print(len(encoder_input_reshaped))	
      for i, id in enumerate(flat_ids):
        # pad the inputs
        id = [id]
        padding_applied.append(MAX_TOKEN_LENGTH-tf.shape(id)[1])    
        id = tf.pad(
          id,  [[0, 0,], [0,MAX_TOKEN_LENGTH-tf.shape(id)[1]]], "CONSTANT")
#         print(encoder_input_reshaped[i])
#         print(id)
        batch_inputs.append([encoder_input_reshaped[i], id])
    
      ds = tf.data.Dataset.from_tensor_slices((batch_inputs)).map(lambda x: {'inputs': x[0], "dec_inputs": x[1]})
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
      ds = ds.with_options(options)
      
      flat_logits = symbols_to_logits_fn.predict(ds)
      new_array = []
      for i, pred in enumerate(flat_logits):
        
        new_array.append(np.delete(flat_logits[i], np.s_[-padding_applied[i]:], 0)[-1])
#         flat_logits[i] = flat_logits[i][:padding_applied[i] + 1]
      
      flat_logits = new_array
#     print("original received:")
#     print(flat_logits)
    logits = tf1.reshape(flat_logits, [batch_size, beam_size, -1])
#     print(logits)
    # Convert logits to normalized log probs
    candidate_log_probs = common_layers.log_prob_from_logits(logits)

    # Multiply the probabilities by the current probabilities of the beam.
    # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
    log_probs = candidate_log_probs + tf1.expand_dims(alive_log_probs, axis=2)

    length_penalty = tf1.pow(((5. + tf1.cast(i + 1, tf1.float32)) / 6.), alpha)

    curr_scores = log_probs / length_penalty
    # Flatten out (beam_size, vocab_size) probs in to a list of possibilities
    flat_curr_scores = tf1.reshape(curr_scores, [-1, beam_size * vocab_size])

    topk_scores, topk_ids = tf1.nn.top_k(flat_curr_scores, k=beam_size * 2)

    # Recovering the log probs because we will need to send them back
    topk_log_probs = topk_scores * length_penalty

    # Work out what beam the top probs are in.
    topk_beam_index = topk_ids // vocab_size
    topk_ids %= vocab_size  # Unflatten the ids

    if not use_tpu:
      # The next three steps are to create coordinates for tf1.gather_nd to pull
      # out the correct sequences from id's that we need to grow.
      # We will also use the coordinates to gather the booleans of the beam
      # items that survived.
      batch_pos = compute_batch_indices(batch_size, beam_size * 2)
      topk_coordinates = tf1.stack([batch_pos, topk_beam_index], axis=2)

      # Gather up the most probable 2*beams both for the ids and
      # finished_in_alive bools
      topk_seq = tf1.gather_nd(alive_seq, topk_coordinates)
      if states:
        states = nest.map_structure(
            lambda state: tf1.gather_nd(state, topk_coordinates), states)

      # Append the most probable alive
      topk_seq = tf1.concat([tf.cast(topk_seq, tf.int64) , tf.cast(tf1.expand_dims(topk_ids, axis=2), tf.int64)], axis=2)
    else:
      # Gather up the most probable 2*beams both for the ids and
      # finished_in_alive bools
      topk_seq = fast_tpu_gather(alive_seq, topk_beam_index)

      if states:
        states = nest.map_structure(
            lambda state: fast_tpu_gather(state, topk_beam_index), states)

      # Update the most probable alive
      topk_seq = tf1.transpose(topk_seq, perm=[2, 0, 1])
      topk_seq = inplace_ops.alias_inplace_update(topk_seq, i + 1, topk_ids)
      topk_seq = tf1.transpose(topk_seq, perm=[1, 2, 0])

#     print(tf.cast(topk_ids, tf.int32))
#     print(tf.cast(eos_id, tf.int32))
#     print(tf1.equal(tf.cast(topk_ids, tf.int32), tf.cast(eos_id, tf.int32)))
    topk_finished = tf1.equal(tf.cast(topk_ids, tf.int32), tf.cast(eos_id, tf.int32))

    return topk_seq, topk_log_probs, topk_scores, topk_finished, states

  def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,	
                 finished_flags, states):
    """Inner beam search loop.
    There are three groups of tensors, alive, finished, and topk.
    The alive group contains information about the current alive sequences
    The topk group contains information about alive + topk current decoded words
    the finished group contains information about finished sentences, that is,
    the ones that have decoded to <EOS>. These are what we return.
    The general beam search algorithm is as follows:
    While we haven't terminated (pls look at termination condition)
      1. Grow the current alive to get beam*2 topk sequences
      2. Among the topk, keep the top beam_size ones that haven't reached EOS
      into alive
      3. Among the topk, keep the top beam_size ones have reached EOS into
      finished
    Repeat
    To make things simple with using fixed size tensors, we will end
    up inserting unfinished sequences into finished in the beginning. To stop
    that we add -ve INF to the score of the unfinished sequence so that when a
    true finished sequence does appear, it will have a higher score than all the
    unfinished ones.
    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_seq: Current finished sequences.
        [batch_size, beam_size, i+1]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      states: dict (possibly nested) of decoding states.
    Returns:
      Tuple of
        (Incremented loop index
         New alive sequences,
         Log probs of the alive sequences,
         New finished sequences,
         Scores of the new finished sequences,
         Flags indicating which sequence in finished as reached EOS,
         dict of final decoding states)
    """

    # Each inner loop, we carry out three steps:
    # 1. Get the current topk items.
    # 2. Extract the ones that have finished and haven't finished
    # 3. Recompute the contents of finished based on scores.
    topk_seq, topk_log_probs, topk_scores, topk_finished, states = grow_topk(	
        i, alive_seq, alive_log_probs, states)
    alive_seq, alive_log_probs, _, states = grow_alive(
        topk_seq, topk_scores, topk_log_probs, topk_finished, states)
    finished_seq, finished_scores, finished_flags, _ = grow_finished(
        finished_seq, finished_scores, finished_flags, topk_seq, topk_scores,
        topk_finished)

    return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
            finished_flags, states)

  def _is_not_finished(i, unused_alive_seq, alive_log_probs,
                       unused_finished_seq, finished_scores,
                       unused_finished_in_finished, unused_states):
    """Checking termination condition.
    We terminate when we decoded up to decode_length or the lowest scoring item
    in finished has a greater score that the highest prob item in alive divided
    by the max length penalty
    Args:
      i: loop index
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
    Returns:
      Bool.
    """
    max_length_penalty = tf1.pow(((5. + tf1.cast(decode_length, tf1.float32)) / 6.), alpha)
    # The best possible score of the most likely alive sequence.
    lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty


  # by taking the max score we only care about the first beam;
  # as soon as this first beam cannot be beaten from the alive beams
  # the beam decoder can stop.
  # similarly to the above, if the top beam is not completed, its
  # finished_score is -INF, thus it will not activate the
  # bound_is_met condition. (i.e., decoder will keep going on).
  # note we need to find the max for every sequence eparately - so, we need
  # to keep the batch dimension (see axis=1)
    lowest_score_of_finished_in_finished = tf1.reduce_max(finished_scores,
                                                       axis=1)

    bound_is_met = tf1.reduce_all(
        tf1.greater(lowest_score_of_finished_in_finished,
                   lower_bound_alive_scores))

    return tf1.logical_and(
        tf1.less(i, decode_length), tf1.logical_not(bound_is_met))

  inner_shape = tf1.TensorShape([None, None, None])

  state_struc = nest.map_structure(get_state_shape_invariants, states)
  (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
    finished_flags, states) = tf1.while_loop(	
       _is_not_finished,	
       inner_loop, [	
           tf1.constant(0), alive_seq, alive_log_probs, finished_seq,	
           finished_scores, finished_flags, states	
       ],	
       shape_invariants=[	
           tf1.TensorShape([]),	
           inner_shape,	
           alive_log_probs.get_shape(),	
           inner_shape,	
           finished_scores.get_shape(),	
           finished_flags.get_shape(),	
           state_struc	
       ],	
       parallel_iterations=1,	
       back_prop=False )

  alive_seq.set_shape((None, beam_size, None))
  finished_seq.set_shape((None, beam_size, None))

  # Accounting for corner case: It's possible that no sequence in alive for a
  # particular batch item ever reached EOS. In that case, we should just copy
  # the contents of alive for that batch item. tf1.reduce_any(finished_flags, 1)
  # if 0, means that no sequence for that batch index had reached EOS. We need
  # to do the same for the scores as well.
  finished_seq = tf1.where(
      tf1.reduce_any(finished_flags, 1), finished_seq, alive_seq)
  finished_scores = tf1.where(
      tf1.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
  return finished_seq, finished_scores, states


# %%
def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

(temp_sent,__), _ = tokenize_pairs([""], tf.constant(" test sent 2 this is a very long sent ence here tesetxc dsaf adsa fgf dsfd f fs")[tf.newaxis])

class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    #@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentences, beam_width=3):
        # input sentence is english, hence adding the start and end token
        for i, sentence in enumerate(sentences):
            assert isinstance(sentence, tf.Tensor)
            if len(sentence.shape) == 0:
                sentence = sentence[tf.newaxis]

            if SRC_LANG == "en" and TGT_LANG == "ic":
                (sentence_tok,__), _ = tokenize_pairs(sentence, [''])
            else:
                (sentence_tok,__), _ = tokenize_pairs([''],sentence)
            sentences[i] = sentence_tok

        #TODO to change this ->
        encoder_input = sentences
        if SRC_LANG == "en" and TGT_LANG == "ic":
            # as the target is maltese, the first token to the transformer should be the
            # maltese start token.
            start_end = self.tokenizers.ic.tokenize([''])[0]
            start = start_end[0][tf.newaxis]
            end = start_end[1][tf.newaxis]
            
            
            vocab_length = self.tokenizers.ic.get_vocab_size()
        else:
            # as the target is english, the first token to the transformer should be the
            # english start token.
            start_end = self.tokenizers.en.tokenize([''])[0]
            start = start_end[0][tf.newaxis]
            end = start_end[1][tf.newaxis]
        
            
            
            vocab_length = self.tokenizers.en.get_vocab_size()

        batch_size = len(encoder_input)
        initial_ids = start * tf.ones([batch_size], dtype=tf.int64)
        decoded_ids, scores, _ = beam_search(
            self.transformer,
            initial_ids,
            beam_width,
            MAX_TOKEN_LENGTH-1,
            vocab_length,
            alpha=0.6,
            states={},
            eos_id=end,
            stop_early=True,
            use_tpu=False,
            use_top_k_with_unique=True, encoder_input=encoder_input)
        
        toReturn = []
        for i, score in enumerate(scores):

            predicted_id = tf.argmax(score, axis=-1)

            output = [decoded_ids[i][predicted_id]]

            if SRC_LANG == "en" and TGT_LANG == "ic":
                text = self.tokenizers.ic.detokenize(output)
    #             tokens = tokenizers.ic.lookup(output)[0]
            else:
                text = self.tokenizers.en.detokenize(output)
    #             tokens = tokenizers.en.lookup(output)[0]
            toReturn.append(text)
        return toReturn
translator = Translator(tokenizers, transformer_test)
ground_truth = "Gurnata tajba"
sentence = "this is a problem we have to solve."





# %%
# %%time
res = translator([tf.constant(sentence), tf.constant(sentence)])
print(res)

# %%
# %%time

ground_truth = "jien qrajt "
sentence = "I read about "

res = translator([tf.constant(ground_truth)])
# print_translation(sentence, translated_text, ground_truth)
print(res)


# %%

class TranslatorForConfidence(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer
    
    def __call__(self, sentence, sentence_tgt):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]


        if SRC_LANG == "en" and TGT_LANG == "ic":
            (sentence,__), _ = tokenize_pairs(sentence, [''])
        else:
            (sentence,__), _ = tokenize_pairs([''],sentence)


        encoder_input = sentence

        if SRC_LANG == "en" and TGT_LANG == "ic":
            # as the target is maltese, the first token to the transformer should be the
            # maltese start token.
            start_end = self.tokenizers.ic.tokenize([sentence_tgt])[0]
            start = start_end[0][tf.newaxis]
            end = start_end[-1][tf.newaxis]
        else:
            # as the target is english, the first token to the transformer should be the
            # english start token.
            start_end = self.tokenizers.en.tokenize([sentence_tgt])[0]
            start = start_end[0][tf.newaxis]
            end = start_end[-1][tf.newaxis]
      


        scores = []
        batch_inputs = []
            
        output_array = tf.TensorArray(dtype=tf.int64, size=MAX_TOKEN_LENGTH, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in range(MAX_TOKEN_LENGTH):
            output = tf.transpose(output_array.stack())
            batch_inputs.append([sentence, output])


            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i+1, [start_end[i+1]])
            if [start_end[i+1]] == end:
                break
        
        output = tf.transpose(output_array.stack())
        ds = tf.data.Dataset.from_tensor_slices((batch_inputs)).map(lambda x: {'inputs': x[0], "dec_inputs": x[1]})
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        total_predictions = transformer_test.predict(ds)      
        
        for i in range(len(total_predictions)):
            if i == 0:
                scores.append(0)
                continue
            scores.append(tf.gather(tf.math.log(tf.nn.softmax(total_predictions[i][i-1])), start_end[i-1])) 
        return scores
    
    
    
ground_truth = "my problem"
sentence = "problema tieghi"
translator_for_confidence = TranslatorForConfidence(tokenizers, transformer_test)    

# %%

res = translator_for_confidence(tf.constant(sentence), tf.constant(ground_truth))
print(res)


# %%
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    #@tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentences):
        for i, sentence in enumerate(sentences):
            sentences[i] = tf.constant(sentence)
        result = self.translator(sentences)

        return result


# In the above `tf.function` only the output sentence is returned. Thanks to the [non-strict execution](https://tensorflow.org/guide/intro_to_graphs) in `tf.function` any unnecessary values are never computed.

# %%


translator_exp = ExportTranslator(translator)


# Since the model is decoding the predictions using `tf.argmax` the predictions are deterministic. The original model and one reloaded from its `SavedModel` should give identical predictions:

# %%


tmp = translator_exp(["I read about triceratops in the book."])
print(tmp)

# %%


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(
      labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


# %%


in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.ic.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.ic.lookup(in_tokens)[0]
# print(in_tokens)


# %%


# print(translated_tokens)
# %%


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    if SRC_LANG == "en" and TGT_LANG == "ic":
        in_tokens = tokenizers.ic.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.ic.lookup(in_tokens)[0]
    else:
        in_tokens = tokenizers.en.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.en.lookup(in_tokens)[0]
    in_tokens

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()

# %%


tmp2 = translator_exp(["I read about triceratops in the book."])
print(tmp2)

# %%
#translator.save('translator')

# tf.saved_model.save(translator_exp, export_dir='translator_exp_' + SRC_LANG + "_" + TGT_LANG)
transformer_test.save_weights('translator_for_confidence_' + SRC_LANG + "_" + TGT_LANG)
# transformer_test.save_weights('translator_exp_' + SRC_LANG + "_" + TGT_LANG)
print('translator_for_confidence_' + SRC_LANG + "_" + TGT_LANG)
print("DONE TRANSFORMER.PY")

# %%
transformer_test.load_weights('./translator_for_confidence_' + SRC_LANG + "_" + TGT_LANG)
translator = Translator(tokenizers, transformer_test)
reloaded = ExportTranslator(translator)
test = reloaded(["I read about triceratops in the book."])
print(test)

# %%
too_big_predictions = []
toWrite = []
data_in = []
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/test.' + SRC_LANG, "r", encoding="utf-8") as f:
    data_in = f.read().split('\n')

lengthOfExamples = len(data_in)
print(lengthOfExamples)


i = 0
batch_size = 25
import os 

if os.path.exists(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG) == False:
    open(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG, "w").close
    
    
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG, "r+", encoding="utf-8") as fp:
    i = len(fp.readlines())
print('Total lines so far:', i)


if os.path.exists(os.path.dirname(os.path.realpath("__file__")) + '/data/tooBigPredictions.' + TGT_LANG) == False:
    open(os.path.dirname(os.path.realpath("__file__")) + '/data/tooBigPredictions.' + TGT_LANG, "w").close
    
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/tooBigPredictions.' + TGT_LANG, "r+", encoding="utf-8") as fp:
    too_big_predictions = len(fp.readlines())
print('Total lines too big so far:', too_big_predictions)

# %%

print("Writing to: " + os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG)
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG, "a+", encoding="utf-8") as w:
    with open(os.path.dirname(os.path.realpath("__file__")) + '/data/tooBigPredictions.' + TGT_LANG, "a+", encoding="utf-8") as wTooBig:
        while (i+too_big_predictions) <= lengthOfExamples-1:
            print(i+too_big_predictions)

            toPredict = []
            for curr_index in range(i+too_big_predictions, i +too_big_predictions + batch_size):
                
                if SRC_LANG == "en":
                    en = tokenizers.en.tokenize([data_in[curr_index]])
                    en = en.to_tensor()
                    if tf.shape(en)[1] >= MAX_TOKEN_LENGTH:
                        print("too big")
                        wTooBig.write(str(curr_index))
                        too_big_predictions += 1
                    else:
                        toPredict.append(data_in[curr_index])
                if SRC_LANG == "ic":
                    ic = tokenizers.ic.tokenize([data_in[curr_index]])
                    ic = ic.to_tensor()
                    if tf.shape(ic)[1] >= MAX_TOKEN_LENGTH:
                        print("too big")
                        wTooBig.write(str(curr_index))
                        too_big_predictions += 1
                    else:
                        toPredict.append(data_in[curr_index])

            print("predicting: ")
            output = reloaded(toPredict)
            res = [el[0].numpy().decode('UTF-8') for el in output]
            w.write("\n".join(res)) 

            # Increment i by the too_big_predictions and by the actual predictions
            i += batch_size


with open(os.path.dirname(os.path.realpath("__file__")) + '/data/tooBigPredictions.' + TGT_LANG, "r+", encoding="utf-8") as fp:
    too_big_predictions = fp.readlines()
    
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/test.' + SRC_LANG, "w", encoding="utf-8") as outfile:
    for pos, line in enumerate(data_en):
        if pos not in too_big_predictions:
            outfile.write(line) 
            outfile.write("\n")  
# with open(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.' + TGT_LANG, "w", encoding="utf-16") as w:
#     for line in toWrite:
#         w.write(line)

# %%
