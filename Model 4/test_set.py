# %%
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install -q tfds-nightly')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install unbabel-comet')
get_ipython().system('pip install jupyter-resource-usage')


# %%
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install tensorflow-model-optimization')
get_ipython().system('pip install -U "tensorflow-text==2.8.*"')

import collections
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from tensorflow import keras
import test_dataset

import tempfile
os.chdir("test_dataset")
print("\n\n LOADING EXAMPLES \n\n\n")

examples, metadata = tfds.load(name="test_dataset", with_info=True,
                               as_supervised=True, download=True, data_dir="test_dataset")
train_examples, val_examples = examples['train'], examples['validation']
model_name = "ted_hrlr_translate_mt_en_converter"

tokenizers = tf.saved_model.load("../" + model_name)

print("\n\n LOADED TOKENIZERS \n\n\n")



# %%
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
# Populate with typical keras callbacks

BATCH_SIZE = 64
# Compute end step to finish pruning after X epochs.
EPOCHS = 10
# Maximum sentence length
MAX_LENGTH = 50

num_images = train_examples.cardinality().numpy()
end_step = np.ceil(num_images / BATCH_SIZE).astype(np.int32) * EPOCHS

# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                             final_sparsity=0.80,
                                                             begin_step=0,
                                                             end_step=end_step),
}

print(num_images)
print(end_step)

# %%

import tensorflow_text as text

def tokenize_pairs(mt, en):
    mt = tokenizers.mt.tokenize(mt)
    en = tokenizers.en.tokenize(en)
    input_word_ids, input_mask = text.pad_model_inputs(
  mt, max_seq_length=MAX_LENGTH-1)
    input_word_ids_en, input_mask = text.pad_model_inputs(
  en, max_seq_length=MAX_LENGTH-1)
    return (input_word_ids, input_word_ids_en), input_word_ids_en


BUFFER_SIZE = 20000


def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


print("\n\n MADE BATCHES n\n")


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)



# %%
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self, position, d_model).__init__(position, d_model, **kwargs)
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
    



print("\n\n LOADING TF LITE MODEL \n\n\n")

# %%
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='quantized_and_pruned.tflite')


print("\n\n ALLOCATING TENSORS \n\n\n")


# %%
interpreter.allocate_tensors()


print("\n\n SIG RUNNER \n\n\n")


# %%
predict = interpreter.get_signature_runner()


# %%
print(interpreter.get_input_details())


# %%
print(interpreter.get_signature_list())

# %%
# sentence = "this is a problem we have to solve."
# empty_sentence = ""
# sentence = tf.constant(sentence)
# empty_sentence = tf.constant(empty_sentence)
# if len(sentence.shape) == 0:
#     sentence = sentence[tf.newaxis]
#     empty_sentence = empty_sentence[tf.newaxis]
    

# mt = tokenizers.mt.tokenize(sentence)
# input_word_ids, input_mask = text.pad_model_inputs(
# mt, max_seq_length=MAX_LENGTH-1)

# %%

# output = predict(dec_inputs=np.float32(input_word_ids), inputs=np.float32(input_word_ids))
# sentence = tf.constant(sentence)
# empty_sentence = tf.constant(empty_sentence)
# if len(sentence.shape) == 0:
#     sentence = sentence[tf.newaxis]
#     empty_sentence = empty_sentence[tf.newaxis]
    

# mt = tokenizers.mt.tokenize(sentence)
# input_word_ids, input_mask = text.pad_model_inputs(
# mt, max_seq_length=MAX_LENGTH-1)

# %%
class Translator(tf.keras.Model):
    def __init__(self, tokenizers):
        super().__init__()
        self.tokenizers = tokenizers

    def __call__(self, sentence, max_length=20):
        assert isinstance(sentence, tf.Tensor)
        
        sentence = tf.constant(sentence)
        empty_sentence = tf.constant("")
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
            empty_sentence = empty_sentence[tf.newaxis]


        # input sentence is English, hence adding the start and end token
        en = tokenizers.en.tokenize(sentence)
        input_word_ids, input_mask = text.pad_model_inputs(
        en, max_seq_length=MAX_LENGTH-1)


        # as the target is Maltese, the first token to the transformer should be the
        # maltese start token.
        start_end = self.tokenizers.mt.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        # output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output = tf.expand_dims(start, 0)

        for i in tf.range(max_length):

#             input_data_2 = np.array(encoder_input, dtype=np.float32)
            predictions = predict(inputs=np.float32(input_word_ids), dec_inputs=np.float32(input_word_ids))["outputs"]
            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)


            # concatenated the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

            if predicted_id == end:
                break


        output = tf.squeeze(output, axis=0)
        # output.shape (1, tokens)
        text_to_return = tokenizers.mt.detokenize([output])[0]  # shape: ()

        tokens = tokenizers.mt.lookup([output])[0]

        return text_to_return, tokens


# Create an instance of this `Translator` class, and try it out a few times:

translator = Translator(tokenizers)


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


ground_truth = "din problema li rridu nsolvu."
sentence = "this is a problem we have to solve."

print("passing to translator...")
translated_text, translated_tokens = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)

ground_truth = "dan l-ewwel ktieb li amilt."
sentence = "this is the first book i've ever done."

translated_text, translated_tokens = translator(
    tf.constant("Measure B4: Recapitalisation to cover capital shortage related to divestment of New HBU and Measure B5: Recapitalisation to cover integration costs"))

translated_text, translated_tokens = translator(
    tf.constant(sentence))
print_translation(sentence, translated_text, ground_truth)



# %%
print("evaluating on test set...")
with open(os.path.dirname(os.path.realpath("__file__")) + '/data/test.en', "r", encoding="utf-16") as f:
    with open(os.path.dirname(os.path.realpath("__file__")) + '/data/predictions.mt', "w", encoding="utf-16") as w:
        data = f.read()
        data_en = data.split('\n')
        lengthOfExamples = len(data_en)
        print(lengthOfExamples)
        
        
        for i in range(lengthOfExamples):
            translated_text, translated_tokens = translator(tf.constant(str(data_en[i])))

            w.write(translated_text.numpy().decode('utf-8'))
            if i < lengthOfExamples - 1:
                w.write("\n")
                
                
                

