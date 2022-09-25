#!/usr/bin/env python
# coding: utf-8
# %%

from __future__ import division

get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install sentence_splitter')
get_ipython().system('pip install -q tfds-nightly')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install jupyter-resource-usage')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install tensorflow-model-optimization')

get_ipython().system('pip install psutil')
get_ipython().system('pip install datasets')
get_ipython().system('pip install tensor2tensor')
get_ipython().system('pip install pympler')
get_ipython().system('pip install tensorflow-addons')
get_ipython().system('pip uninstall -y tensorflow-text')
get_ipython().system('pip install tensorflow-text')
get_ipython().system('pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116')




# %%


from itertools import repeat
import torch
from transformers import AutoModel, AutoTokenizer
import datasets
from tokenisemt import MTWordTokenizer
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import sys, codecs, numpy
import tensorflow.compat.v1 as tf1
from tensor2tensor.layers import common_layers
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest

import tensorflow_text as text
tok = MTWordTokenizer()


class autovivify_list(dict):
    """A pickleable version of collections.defaultdict"""

    def __missing__(self, key):
        """Given a missing key, set initial value to an empty list"""
        value = self[key] = []
        return value

    def __add__(self, x):
        """Override addition for numeric types when self is empty"""
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        """Also provide subtraction method"""
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError


# def build_word_vector_matrix(vector_file, n_words):


def find_word_clusters(labels_array, cluster_labels):
    """ Return the set of words in each cluster"""
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words



# %%


from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import math
model_name = "MLRS/mBERTu"
model, tokeniser = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
# set device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# put model on GPU (is done in-place)
model.to(device)
# get_ipython().run_line_magic('cd', 'test_dataset')
os.chdir("dataset_without_masks")



# %%

torch.cuda.empty_cache()
def get_embeddings(model, tokeniser, sentences):
    try:
        input = tokeniser(sentences,
                                            max_length=model.embeddings.position_embeddings.num_embeddings,
                                            padding="max_length",
                                            truncation=True,
                                            # return_overflowing_tokens=True,
                                            return_tensors="pt",
                                            )
        input.to(device)
        output = model(**input)


        vocabulary = dict(zip(tokeniser.get_vocab().values(), tokeniser.get_vocab().keys()))
        toReturn = []
        len_sentences = len(sentences)
        for numOfInputs in range(len_sentences):
            words = []
            embeddings = []

            for i, token in enumerate(map(lambda token_id: vocabulary[token_id],
                                          [instance for instance in input["input_ids"][numOfInputs].tolist() 
                                           ])):
                try:
                    embedding = output[0][:, i]
                    if token in ("[CLS]", "[SEP]", "[PAD]"):
                        continue
                    elif token.startswith("##"):
                        words[-1] += token[2:]
                        embeddings[-1] = torch.cat((embeddings[-1], embedding), dim=0)
                    # removes symbols only such as '-'
                    elif (all(not c.isalnum() for c in token)) or (all(c.isdigit() for c in token)):
                        continue
                    else:
                        words.append(token)
                        embeddings.append(embedding)
                except:
                    print(words)
                    print(token)
            toReturn.append({"words": words, "embeddings": [embedding.mean(dim=0) for embedding in embeddings]})
        del input, model, sentences, tokeniser, len_sentences, vocabulary
        torch.cuda.empty_cache()
        return toReturn 
    except:
        try:
            del input, model, sentences, tokeniser, len_sentences, vocabulary
            torch.cuda.empty_cache()
        except:
            print("")
# torch.cuda.empty_cache()
# temp_sents = ['20 Kwota tat-tariffa [4]', "32011 R 0432: Ir-Regolament tal-Kummissjoni (UE) Nru 432/2011 tal-4 ta' Mejju 2011 li ma jagħtix kunsens li jiġu awtorizzati ċerti stqarrijiet li jirrigwardaw l-effetti tal-prodotti tal-ikel fuq is-saħħa, ħlief dawk li jirreferu għal tnaqqis ta' riskju ta' mard u ta' riskju għall-iżvilupp u għas-saħħa tat-tfal (ĠU L 115, 5.5.2011, p. 1)."]
# torch.cuda.empty_cache()
# res = get_embeddings(model, tokeniser, temp_sents)



# %%
os.system("tfds build")

# %%


examples, metadata = tfds.load(name="dataset_without_masks", with_info=True,
                               as_supervised=True, download=True)
train_examples, val_examples = examples['train'], examples['validation']


BATCH_SIZE = 256
# Maximum sentence length
MAX_LENGTH = 80
BUFFER_SIZE = 20000


def tokenize_pairs(en, mt):
    return en, mt




def make_batches(ds):
    return (
        ds
            .cache()
#             .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)


# # Step 1: Getting a list of multi-sense words

# %%
total_words_en = []
total_words_mt = []
total_embeddings_en = []
total_embeddings_mt = []

monomt = open("../monolingual_data/mono.mt", 'r', encoding="utf-8")
mt_examples = monomt.readlines()
for i, word in enumerate(mt_examples):
    mt_examples[i] = word.replace("\n", "")
# en_examples = datasets.load_dataset("MLRS/korpus_malti", "shuffled", split="train")
en_examples = datasets.load_dataset("wikipedia", "20220301.en")


# %%
# import asyncio

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

#     return wrapped

# @background
def write_embeddings(embeddings_en_file, embedding, total_embeddings_mt):
    with open(embeddings_en_file, "a+", encoding="utf-8") as embeddings_en_file:
        embeddings_en_file.write("\t".join(str(x) for x in embedding.tolist()) + "\n")
        total_embeddings_mt.append(embedding.tolist())
    
    
# @background
def write_words(words_en_file, word, i,j, total_words):
    with open(words_en_file, "a+", encoding="utf-8") as words_en_file:
        words_en_file.write(f"{word}\t{i}_{j}\n")
        total_words.append(word)


# %%
import time
BATCH_SIZE = 20
with open("../words_mt.tsv", "w+", encoding="utf-8") as words_mt_file, \
        open("../words_en.tsv", "w+", encoding="utf-8") as words_en_file:
    words_en_file.write("word\tindex\n")
    words_mt_file.write("word\tindex\n")
    
# total_length = len(en_examples["train"])
# length_sentences_en = math.ceil(total_length*0.000001)
# # length_sentences_en = 20
# print(length_sentences_en)
# print(math.ceil((length_sentences_en - 1) / BATCH_SIZE))
# for i in range(math.ceil((length_sentences_en - 1) / BATCH_SIZE)):
#     print(i)
#     torch.cuda.empty_cache() # PyTorch thing
# #         print(torch.cuda.memory_allocated())
#     with torch.no_grad():
#         print(i*BATCH_SIZE)
#         if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_en:
#             sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):math.ceil(length_sentences_en)]
#         else:
#             sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

#         for j, sentence in enumerate(sentence_en):
#             sentence_en[j] = sentence.replace("\n", " ").strip()
        
#         res = get_embeddings(model, tokeniser, sentence_en)
#         if res is None:
#             continue
#         for i, inp in enumerate(res):
#             words = inp["words"]
#             embeddings = inp["embeddings"]
# #                 assert len(words) == len(embeddings)
#             for embedding in embeddings:
#                 write_embeddings("../embeddings_en.tsv", embedding, total_embeddings_en)
#             for j, word in enumerate(words):
#                 write_words("../words_en.tsv", word, i,j, total_words_en)

#     del sentence_en, res, words, embeddings




length_sentences_mt = math.ceil(len(mt_examples)*0.1)
print("======================MT======================")
print(length_sentences_mt)
for i in range(math.ceil((length_sentences_mt - 1) / BATCH_SIZE)):
    torch.cuda.empty_cache() # PyTorch thing
    with torch.no_grad():
        print(i*BATCH_SIZE)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_mt:
            sentence_mt = mt_examples[(i*BATCH_SIZE):math.ceil(length_sentences_mt)]
        else:
            sentence_mt = mt_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

        for j, sentence in enumerate(sentence_mt):
            sentence_mt[j] = sentence.replace("\n", " ").strip()
        temp_sent = sentence_mt
        res = get_embeddings(model, tokeniser, sentence_mt)
        if res is None:
            continue
        for inp in res:
            words = inp["words"]
            embeddings = inp["embeddings"]
            assert len(words) == len(embeddings)

            for embedding in embeddings:
                write_embeddings("../embeddings_mt.tsv", embedding, total_embeddings_mt)
            for j, word in enumerate(words):
                write_words("../words_mt.tsv", word, i,j, total_words_mt)



# %%

alreadySorted = []
with open("../listOfMultiSenseWords_mt.txt", "w+", encoding="utf-8") as multisensewords_mt:
    
#     for i, word in enumerate(total_words_en):
#         embeddingsOfCurrentWord = []

#         if word in alreadySorted:
#             continue
#         alreadySorted.append(word)

#         embeddingsOfCurrentWord.append(total_embeddings_en[i])

#         # Cet all embeddings of the same word in diff sentences
#         for j, futureWord in enumerate(total_words_en[i+1:]):
#             if futureWord == word:
#                 embeddingsOfCurrentWord.append(total_embeddings_en[j])

#         n_words = len(embeddingsOfCurrentWord)
#         if(n_words < 2):
#             continue

#         eps = 4.4
#         min_samples = 2
#         ret = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddingsOfCurrentWord)

#         if len(set(ret)) > 1 or (len(set(ret)) == 1 and ret[0] == -1):
#             print(word)
#             multisensewords_en.write(word)
#             multisensewords_en.write("\n")
            
            
    print("MT: ")
    print(len(total_words_mt))
    print(len(total_embeddings_mt))
    for i, word in enumerate(total_words_mt):
        embeddingsOfCurrentWord = []

        if word in alreadySorted:
            continue
        alreadySorted.append(word)

        embeddingsOfCurrentWord.append(total_embeddings_mt[i])

        # Cet all embeddings of the same word in diff sentences
        for j, futureWord in enumerate(total_words_mt[i+1:]):
            if futureWord == word:
                embeddingsOfCurrentWord.append(total_embeddings_mt[j])

        n_words = len(embeddingsOfCurrentWord)
        if(n_words < 2):
            continue

        eps = 3.4
        min_samples = 3
        ret = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddingsOfCurrentWord)

        if len(set(ret)) > 2:
            print(word)
            print(ret)
            multisensewords_mt.write(word)
            multisensewords_mt.write("\n")




# %%

# %% [markdown]
#
# # Step 2:  Mining multisense words
#
# <!-- Loop each word in each sentence of OUR monolingual corpus and get the embeddings and save them to file -->

# %%

# length_sentences_mt = len(mt_monolingual_corpus)*0.001
# print(length_sentences_mt)
# for i in range(math.ceil((length_sentences_mt - 1) / BATCH_SIZE)):
#     torch.cuda.empty_cache() # PyTorch thing
#     with torch.no_grad():
#         total_embeddings = []
#         total_words = []
#         print(i*BATCH_SIZE)
#         if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_mt:
#             sentence_mt = mt_monolingual_corpus[(i*BATCH_SIZE):math.ceil(length_sentences_mt)]
#         else:
#             sentence_mt = mt_monolingual_corpus[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]


#         for j, sentence in enumerate(sentence_mt):
#             sentence_mt[j] = sentence.replace("\n", " ").strip()
#         res = get_embeddings(model, tokeniser, sentence_mt)
#         if res != [] and res != None:
#             try:
#                 for i, inp in enumerate(res):
#                     words = inp["words"]
#                     embeddings = inp["embeddings"]
#                     assert len(words) == len(embeddings)
#                     for embedding in embeddings:
#                         write_embeddings("../embeddings_mt_step2.tsv", embedding, total_embeddings)                    

#                     for j, word in enumerate(words):
#                         write_words("../words_mt_step2.tsv", word, i,j, total_words)
#             except:
#                 print("ERROR CAUGHT")
#                 print(res)

#     del sentence_mt, res, total_words, total_embeddings

        

# %% [markdown]
# Traverse each word in each sentence and check for multisense words. If one is found, get the embedding.

# %%

#print(mt_multisense_words)

# found_multisense_en_info = []
found_multisense_mt_info = []

with open("../listOfMultiSenseWords_mt.txt", "r", encoding="utf-8") as multisense_mt:
    
#     en_multisense_words = multisense_en.readlines()
#     for i, word in enumerate(en_multisense_words):
#         en_multisense_words[i] = word.replace("\n", "")
    mt_multisense_words = multisense_mt.readlines()
    for i, word in enumerate(mt_multisense_words):
        mt_multisense_words[i] = word.replace("\n", "")
#     print(en_multisense_words)
    print(mt_multisense_words)
    
    
#     length_sentences_en = len(en_examples["train"])*0.000001     
    
#     print(length_sentences_en)
#     for i in range(math.ceil((length_sentences_en - 1) / BATCH_SIZE)):
     
#         torch.cuda.empty_cache() # PyTorch thing
#         with torch.no_grad():
#             toPopIndexes = []
#             total_embeddings = []
#             total_words = []
#             print(i*BATCH_SIZE)
#             if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_en:
#                 sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):math.ceil(length_sentences_en)]
#             else:
#                 sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

                
#             for j, sentence in enumerate(sentence_en):
#                 sentence_en[j] = sentence.replace("\n", " ").strip()

#             res = get_embeddings(model, tokeniser, sentence_en)
#             if res != [] and res != None:
#                 try:
#                     for i, inp in enumerate(res):
#                         words = inp["words"]
#                         embeddings = inp["embeddings"]

#                         assert len(words) == len(embeddings)
#                         for multisense_word_pos, word in enumerate(sentence_en[i].split()):
#                             if word in en_multisense_words:
#                                 found_multisense_en_info.append({'word': word, 'embedding': embeddings[multisense_word_pos], 'sentence': sentence_en[i]})

#                 except:
#                     print("ERROR CAUGHT2")
#                     print(res)

#             del sentence_en, res, total_embeddings, total_words 
        
   
    length_sentences_mt = len(mt_examples)*0.1     
    for i in range(math.ceil((length_sentences_mt - 1) / BATCH_SIZE)):
        torch.cuda.empty_cache() # PyTorch thing
        with torch.no_grad():
            total_embeddings = []
            total_words = []
            print(i*BATCH_SIZE)
            if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_mt:
                sentence_mt = mt_examples[(i*BATCH_SIZE):math.ceil(length_sentences_mt)]
            else:
                sentence_mt = mt_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

                
            for j, sentence in enumerate(sentence_mt):
                sentence_mt[j] = sentence.replace("\n", " ").strip()
            res = get_embeddings(model, tokeniser, sentence_mt)
            if res != [] and res != None:
                try:
                    for i, inp in enumerate(res):
                        words = inp["words"]
                        embeddings = inp["embeddings"]

                        assert len(words) == len(embeddings)
                        for multisense_word_pos, word in enumerate(words):
                            if word in mt_multisense_words:
                                found_multisense_mt_info.append({'word': word, 'embedding': embeddings[multisense_word_pos], 'sentence': sentence_mt[i]})

                except:
                    print("ERROR3")
                    print(res)

            del sentence_mt, res, total_embeddings, total_words  
            



# %% [markdown]
# Loop the whole target corpus to get the embeddings of the closest words on the target side

# %%


en_embeddings = []
en_words = []
# mt_embeddings = []
# mt_words = []



total_length = len(en_examples["train"])
length_sentences_en = math.ceil(len(en_examples["train"])*0.001)

print(length_sentences_en)
print(math.ceil((length_sentences_en - 1) / BATCH_SIZE))
for i in range(math.ceil((length_sentences_en - 1) / BATCH_SIZE)):
    torch.cuda.empty_cache() # PyTorch thing
#         print(torch.cuda.memory_allocated())
    with torch.no_grad():
        print(i*BATCH_SIZE)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_en:
            sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):math.ceil(length_sentences_en)]
        else:
            sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

        for j, sentence in enumerate(sentence_en):
            sentence_en[j] = sentence.replace("\n", " ").strip()
        print("Getting embeddings")
        res = get_embeddings(model, tokeniser, sentence_en)
        print("Done")
        if res != [] and res != None:
            try:
                for i, inp in enumerate(res):
                    words = inp["words"]
                    embeddings = inp["embeddings"]
                    en_embeddings.append(embeddings)
                    en_words.append(words)
                del sentence_en, res, words, embeddings
            except:
                print(res)
    
    
    
    

# length_sentences_mt = math.ceil(len(mt_examples)*0.000001)
# print(length_sentences_mt)
# for i in range(math.ceil((length_sentences_mt - 1) / BATCH_SIZE)):
#     torch.cuda.empty_cache() # PyTorch thing
#     with torch.no_grad():
#         print(i*BATCH_SIZE)
#         if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_mt:
#             sentence_mt = mt_examples[(i*BATCH_SIZE):math.ceil(length_sentences_mt)]["text"]
#         else:
#             sentence_mt = mt_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]["text"]

#         for j, sentence in enumerate(sentence_mt):
#             sentence_mt[j] = sentence.replace("\n", " ").strip()
#         temp_sent = sentence_mt
#         res = get_embeddings(model, tokeniser, sentence_mt)
#         if res != [] and res != None:
#             try:
#                 for i, inp in enumerate(res):
#                     words = inp["words"]
#                     embeddings = inp["embeddings"]
#                     mt_embeddings.append(embeddings)
#                     mt_words.append(words)
#             except:
#                 print(res)
                



# %% [markdown]
#  Calculate cosine similarity of source multisense word and each target word in each sentence just retrieved above, 
# to see which is closest

# %%
#print(found_multisense_en_info)
#print(found_multisense_mt_info)
# TODO see if this is correct, it was e.detach().numpy() for e in ...
# embeddings_en = [e for e in en_embeddings]



# print("FOUND MULTISENSE EN INFO: ")
# print(found_multisense_en_info)

# print("EMBEDDINGS MT: ")
# print(mt_embeddings)
# results_en = []
# for en_multi in found_multisense_en_info:
#     list_of_sim = []
#     list_of_sim_info = []
#     for i, sentence_embedding in enumerate(mt_embeddings):
#         embeddings_mt = [e for e in sentence_embedding]
#         for j, word_embedding in enumerate(embeddings_mt):
#             res = cosine_similarity(word_embedding.cpu().reshape(1, -1) , en_multi['embedding'].cpu().reshape(1, -1))
#             list_of_sim.append(res)
#             list_of_sim_info.append({'multi_sense_word': en_multi['word'], 'multi_sense_sentence': en_multi['sentence'], 'target_word': mt_words[i][j], 'target_sentence': mt_words[i]})
#     list_of_biggest_sim_indices = sorted(range(len(list_of_sim)), key=lambda i: list_of_sim[i])[-5:]
#     for index in list_of_biggest_sim_indices:
#         results_en.append(list_of_sim_info[index])
        
# print("FINISHED EN")        
results_mt = []
print(len(found_multisense_mt_info))
for i, mt_multi in enumerate(found_multisense_mt_info):
    if( i % 10 == 0):
        print(i)
    list_of_sim = []
    list_of_sim_info = []
    for i, sentence_embedding in enumerate(en_embeddings):
        embeddings_en = [e for e in sentence_embedding]
        for j, word_embedding in enumerate(embeddings_en):
            res = cosine_similarity(word_embedding.cpu().reshape(1, -1) , mt_multi['embedding'].cpu().reshape(1, -1))
            list_of_sim.append(res)
            list_of_sim_info.append({'multi_sense_word': mt_multi['word'], 'multi_sense_sentence': mt_multi['sentence'], 'target_word': en_words[i][j], 'target_sentence': en_words[i]})
    list_of_biggest_sim_indices = sorted(range(len(list_of_sim)), key=lambda i: list_of_sim[i])[-5:]
    for index in list_of_biggest_sim_indices:
        results_mt.append(list_of_sim_info[index])

print(results_en)
print(results_mt)



# 

# %% [markdown]
# ## Step 3: Create sentence pairs for each multisense word and create data to be backtranslated

# %%

# %%

# %%

# %%


# %%


# %%


# %%

# %%
