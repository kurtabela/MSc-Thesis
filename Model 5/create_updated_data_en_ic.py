#!/usr/bin/env python
# coding: utf-8
# %%

from __future__ import division

get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install apache_beam')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install sentence_splitter')
get_ipython().system('pip install -q tfds-nightly')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install sacrebleu')
get_ipython().system('pip install unbabel-comet')
get_ipython().system('pip install jupyter-resource-usage')
get_ipython().system('pip install tensorflow_datasets')
get_ipython().system('pip install tensorflow_text')
get_ipython().system('pip install tokenizers')
get_ipython().system('pip install datasets')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install tensorflow-model-optimization')
get_ipython().system('pip install tensorflow-text')
get_ipython().system('pip install -U huggingface_hub')


get_ipython().system('pip install tokenizer')

# %%



from itertools import repeat
import torch
from transformers import AutoModel, AutoTokenizer
import datasets
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import sys, codecs, numpy


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
model_name = "bert-base-multilingual-cased"
model, tokeniser = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
# set device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# put model on GPU (is done in-place)
model.to(device)
# get_ipython().run_line_magic('cd', 'test_dataset')
os.chdir("dataset_without_masks_ic")



# %%

torch.cuda.empty_cache()
def get_embeddings(model, tokeniser, sentences):

    input = tokeniser(sentences,
                                        max_length=model.embeddings.position_embeddings.num_embeddings,
                                        padding="max_length",
                                        truncation=True,
                                        # return_overflowing_tokens=True,
                                        return_tensors="pt",
                                        )
    #print(device)
    input.to('cuda:0')
    model.to('cuda:0')
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

torch.cuda.empty_cache()
temp_sents = ['20 Kwota tat-tariffa [4]']
torch.cuda.empty_cache()
res = get_embeddings(model, tokeniser, temp_sents)


# %%

#os.system("tfds build")

# %%




BATCH_SIZE = 256
# Maximum sentence length
MAX_LENGTH = 80


# # Step 1: Getting a list of multi-sense words

# %%
total_words_en = []
total_words_ic = []
total_embeddings_en = []
total_embeddings_ic = []
monoic = open("../monolingual_data/mono.ic", 'r', encoding="utf-8")
ic_examples = monoic.readlines()
for i, word in enumerate(ic_examples):
    ic_examples[i] = word.replace("\n", "")

print("READ IC DATA")


# %%
# en_examples = datasets.load_dataset("MLRS/korpus_malti", "shuffled", split="train")
en_examples = datasets.load_dataset("wikipedia", "20220301.en")
print("READ EN DATA")


# %%
# import asyncio

# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

#     return wrapped

# @background
def write_embeddings(embeddings_en_file, embedding, total_embeddings_en):
    with open(embeddings_en_file, "a+", encoding="utf-8") as embeddings_en_file:
        embeddings_en_file.write("\t".join(str(x) for x in embedding.tolist()) + "\n")
        total_embeddings_en.append(embedding.tolist())
    
    
# # @background
def write_words(words_en_file, word, i,j, total_words_en):
    with open(words_en_file, "a+", encoding="utf-8") as words_en_file:
        words_en_file.write(f"{word}\t{i}_{j}\n")
        total_words_en.append(word)


# %%
import time
print("starting.. writing to file")
BATCH_SIZE = 100
# with open("../words_ic.tsv", "w+", encoding="utf-8") as words_ic_file, \
#         open("../words_en_ic.tsv", "w+", encoding="utf-8") as words_en_file:
#     words_en_file.write("word\tindex\n")
#     words_ic_file.write("word\tindex\n")
    



length_sentences_ic = math.ceil(len(ic_examples)*0.005)
print(length_sentences_ic)
#with open("../embeddings_ic.tsv", "a+", encoding="utf-8") as embeddings_en_file, open("../words_ic.tsv", "a+", encoding="utf-8") as words_en_file:
for i in range(math.ceil((length_sentences_ic - 1) / BATCH_SIZE)):
    torch.cuda.empty_cache() # PyTorch thing
    with torch.no_grad():
        print(i*BATCH_SIZE)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_ic:
            sentence_ic = ic_examples[(i*BATCH_SIZE):math.ceil(length_sentences_ic)]
        else:
            sentence_ic = ic_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

        for j, sentence in enumerate(sentence_ic):
            sentence_ic[j] = sentence.replace("\n", " ").strip()
        res = get_embeddings(model, tokeniser, sentence_ic)
        if res is not None:
            for inp in res:
                for embedding in inp["embeddings"]:
                    total_embeddings_ic.append(embedding.tolist())
                for word in inp["words"]:
                    total_words_ic.append(word)

# %%

alreadySorted = []

with open("../listOfMultiSenseWords_ic.txt", "w+", encoding="utf-8") as multisensewords_ic:
            
    print("looping total_words_ic: ")
    print(len(total_words_ic) )
    for i, word in enumerate(total_words_ic):
        
        if( i % 1000 == 0):
            print(i)
        embeddingsOfCurrentWord = []

        if word in alreadySorted:
            continue
        alreadySorted.append(word)

        embeddingsOfCurrentWord.append(total_embeddings_ic[i])

        # Cet all embeddings of the same word in diff sentences
        for j, futureWord in enumerate(total_words_ic[i+1:]):
            if futureWord == word:
                embeddingsOfCurrentWord.append(total_embeddings_ic[j])

        n_words = len(embeddingsOfCurrentWord)
        if(n_words < 2):
            continue

        eps = 4.4
        min_samples = 2
        ret = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddingsOfCurrentWord)

        if -1 in ret:
            length_to_comp = 2
        else:
            length_to_comp = 1
        if len(set(ret)) > length_to_comp:
            print(word)
            multisensewords_ic.write(word)
            multisensewords_ic.write("\n")

del total_words_ic, total_embeddings_ic


# %% [markdown]
#
# # Step 2:  Mining multisense words
#
# Loop each word in each sentence of OUR monolingual corpus and get the embeddings and save them to file

# %%
monoic = open("../monolingual_data/mono.ic", 'r', encoding="utf-8")
ic_monolingual_corpus = monoic.readlines()



import tensorflow_text as text

with open("../embeddings_ic_step2.tsv", "w+", encoding="utf-8") as embeddingsic_file, \
        open("../words_ic_step2.tsv", "w+", encoding="utf-8") as wordsic_file:
    print("files created")

# %%



length_sentences_ic = len(ic_monolingual_corpus)*0.015
print(math.ceil((length_sentences_ic - 1) / BATCH_SIZE))
for i in range(math.ceil((length_sentences_ic - 1) / BATCH_SIZE)):
    torch.cuda.empty_cache() # PyTorch thing
    with torch.no_grad():
        total_embeddings = []
        total_words = []
        if( i % 10 == 0):
            print(i)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_ic:
            sentence_ic = ic_monolingual_corpus[(i*BATCH_SIZE):math.ceil(length_sentences_ic)]
        else:
            sentence_ic = ic_monolingual_corpus[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]


        for j, sentence in enumerate(sentence_ic):
            sentence_ic[j] = sentence.replace("\n", " ").strip()
        res = get_embeddings(model, tokeniser, sentence_ic)
        if res != [] and res != None:
#             try:
            for i, inp in enumerate(res):
                words = inp["words"]
                embeddings = inp["embeddings"]
                for embedding in inp["embeddings"]:
                    total_embeddings.append(embedding.tolist())
                for word in inp["words"]:
                    total_words.append(word)
#             except:
#                 print("ERRORRR")

    del sentence_ic, res, total_words, total_embeddings

        




# %% [markdown]
# Traverse each word in each sentence and check for multisense words. If one is found, get the embedding.

# %%


# found_multisense_en_info = []
found_multisense_ic_info = []

with open("../listOfMultiSenseWords_ic.txt", "r", encoding="utf-8") as multisense_ic:

    ic_multisense_words = multisense_ic.readlines()
    for i, word in enumerate(ic_multisense_words):
        ic_multisense_words[i] = word.replace("\n", "").strip().lower()
    
    print("MULTI SENSE WORDS FOUND: ")
    print(ic_multisense_words)
#     print(en_multisense_words)
    print(ic_multisense_words)
    
    

   
    length_sentences_ic = len(ic_examples)*0.0025   
    print(math.ceil((length_sentences_ic - 1) / BATCH_SIZE))
    for i in range(math.ceil((length_sentences_ic - 1) / BATCH_SIZE)):
        torch.cuda.empty_cache() # PyTorch thing
        with torch.no_grad():
            total_embeddings = []
            total_words = []
            print(i)
            if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_ic:
                sentence_ic = ic_examples[(i*BATCH_SIZE):math.ceil(length_sentences_ic)]
            else:
                sentence_ic = ic_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

                
            for j, sentence in enumerate(sentence_ic):
                sentence_ic[j] = sentence.replace("\n", " ").strip()
            res = get_embeddings(model, tokeniser, sentence_ic)
            if res != [] and res != None:
                try:
                    for i, inp in enumerate(res):
                        words = inp["words"]
                        embeddings = inp["embeddings"]

                        assert len(words) == len(embeddings)
                        for multisense_word_pos, word in enumerate(words):
                            if word.replace("\n", "").strip().lower() in ic_multisense_words:
                                found_multisense_ic_info.append({'word': word, 'embedding': embeddings[multisense_word_pos], 'sentence': sentence_ic[i]})

                except:
                    print(res)

            del sentence_ic, res, total_embeddings, total_words  
            



# %%
print(len(found_multisense_ic_info))

# %% [markdown]
# Loop the whole target corpus to get the embeddings of the closest words on the target side

# %%
torch.cuda.current_device()

# %%
print("Loop the whole target corpus to get the embeddings of the closest words on the target side")

en_embeddings = []
en_words = []
# ic_embeddings = []
# ic_words = []



length_sentences_en = math.ceil(len(en_examples["train"])*0.0001)
print(length_sentences_en)
print(math.ceil((length_sentences_en - 1) / BATCH_SIZE))



# %%


BATCH_SIZE = 10
for i in range(math.ceil((length_sentences_en - 1) / BATCH_SIZE)):
    print(i)
    torch.cuda.empty_cache() # PyTorch thing
    with torch.no_grad():
    #         print(torch.cuda.memory_allocated())
        print(i*BATCH_SIZE)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_en:
            sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):math.ceil(length_sentences_en)]
        else:
            sentence_en = en_examples["train"]["text"][(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

        for j, sentence in enumerate(sentence_en):
            sentence_en[j] = sentence.replace("\n", " ").strip()
        print("getting embeddings")
        res = get_embeddings(model, tokeniser, sentence_en)
        print("done")
        if res != [] and res != None:
            try:
                for i, inp in enumerate(res):
                    words = inp["words"]
                    embeddings = inp["embeddings"]
                    en_embeddings.append(embeddings)
                    en_words.append(words)
            except:
                print(res)
    #         break
    


# %% [markdown]
# Calculate cosine similarity of source multisense word and each target word in each sentence just retrieved above, 
# to see which is closest
#
#
#         
#

# %%
# import dill
# dill.dump_session('notebook_env.db')


# %%
# import dill
# dill.load_session('notebook_env.db')

# %%
from sklearn.neighbors import NearestNeighbors
import copy
model_nn = NearestNeighbors(n_neighbors=1,
                         metric='cosine',
                         algorithm='brute',
                         n_jobs=-1)
embeddings_to_be_mapped = []
embeddings_info_to_be_mapped = []
for i, sentence_embedding in enumerate(en_embeddings):
    embeddings_en = [e for e in sentence_embedding]
    for j, word_embedding in enumerate(embeddings_en):
        embeddings_to_be_mapped.append(word_embedding.cpu().reshape(1, -1)[0].numpy())
        embeddings_info_to_be_mapped.append({'target_word': en_words[i][j], 'target_sentence': en_words[i]})

# %%
print(len(embeddings_to_be_mapped))

# %%

model_nn.fit(embeddings_to_be_mapped)

# %%

results_ic = []
print(len(found_multisense_ic_info))
for i, ic_multi in enumerate(found_multisense_ic_info):
    if( i % 100 == 0):
        print(i)
    list_of_sim = []
    list_of_sim_info = []
    dist, results = model_nn.kneighbors([ic_multi['embedding'].cpu().reshape(1, -1).numpy()[0]])
    for res in results[0]:
        results_ic.append({'multi_sense_word': ic_multi['word'], 'multi_sense_sentence': ic_multi['sentence'], 'target_word': embeddings_info_to_be_mapped[res]['target_word'], 'target_sentence':  embeddings_info_to_be_mapped[res]['target_sentence']})
        

print(results_ic)

# %%
import json
with open("results_ic.txt", "w+", encoding="utf-8") as results_ic_file:
    json.dump(results_ic, results_ic_file)
    #results_ic_file.write("\n".join(json.stringify(results_ic)))

# %%

# %%
