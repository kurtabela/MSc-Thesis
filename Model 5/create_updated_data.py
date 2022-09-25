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

get_ipython().system('pip uninstall -y transformers')
get_ipython().system('pip uninstall -y huggingface_hub')
get_ipython().system('pip install -U transformers')
get_ipython().system('pip install -U huggingface_hub')




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
os.chdir("datasetwithoutmasks")



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
total_words_en = []
total_words_mt = []
total_embeddings_en = []
total_embeddings_mt = []

monomt = open("../monolingual_data/mono.mt", 'r', encoding="utf-8")
mt_examples = monomt.readlines()
#mt_examples = mt_examples[:10000]
for i, word in enumerate(mt_examples):
    mt_examples[i] = word.replace("\n", "")
    
print("MT IS READ")


# %%
# en_examples = datasets.load_dataset("MLRS/korpus_malti", "shuffled", split="train")
en_examples = datasets.load_dataset("wikipedia", "20220301.en")
print("Done reading examples")

# %%

# def write_embeddings(embeddings_en_file, embedding, total_embeddings_mt):
    
        
    
    
# def write_words(words_en_file, word, i,j, total_words):
#     with open(words_en_file, "a+", encoding="utf-8") as words_en_file:


# %%
import time
BATCH_SIZE = 20
# with open("../words_mt.tsv", "w+", encoding="utf-8") as words_mt_file, \
#         open("../words_en.tsv", "w+", encoding="utf-8") as words_en_file:
#     words_en_file.write("word\tindex\n")
#     words_mt_file.write("word\tindex\n")


length_sentences_mt = math.ceil(len(mt_examples)*0.0001)
print("======================MT======================")
print(length_sentences_mt)
# with open("../embeddings_mt.tsv", "a+", encoding="utf-8") as embeddings_en_file, open("../words_mt.tsv", "a+", encoding="utf-8") as words_en_file:
for i in range(math.ceil((length_sentences_mt - 1) / BATCH_SIZE)):
    with torch.no_grad():
        print(i*BATCH_SIZE)
        if (i*BATCH_SIZE)+BATCH_SIZE >= length_sentences_mt:
            sentence_mt = mt_examples[(i*BATCH_SIZE):math.ceil(length_sentences_mt)]
        else:
            sentence_mt = mt_examples[(i*BATCH_SIZE):(i*BATCH_SIZE)+BATCH_SIZE]

        for j, sentence in enumerate(sentence_mt):
            sentence_mt[j] = sentence.replace("\n", " ").strip()
        res = get_embeddings(model, tokeniser, sentence_mt)
        if res is not None:
            for inp in res:
                for embedding in inp["embeddings"]:
                    total_embeddings_mt.append(embedding.tolist())
                for word in inp["words"]:
                    total_words_mt.append(word)



# %%

alreadySorted = []

with open("../listOfMultiSenseWords_mt.txt", "w+", encoding="utf-8") as multisensewords_mt:
            
    print("looping total_words_mt: ")
    print(len(total_words_mt) )
    for i, word in enumerate(total_words_mt):
        
        if( i % 1000 == 0):
            print(i)
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

        eps = 5.1
        min_samples = 2
        ret = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddingsOfCurrentWord)

        if -1 in ret:
            length_to_comp = 3
        else:
            length_to_comp = 2
        if len(set(ret)) > length_to_comp:
            multisensewords_mt.write(word)
            multisensewords_mt.write("\n")




del total_words_mt, total_embeddings_mt

# %% [markdown]
#
# # Step 2:  Mining multisense words
#
# <!-- Loop each word in each sentence of OUR monolingual corpus and get the embeddings and save them to file -->

# %% [markdown]
# Traverse each word in each sentence and check for multisense words. If one is found, get the embedding.

# %%

found_multisense_mt_info = []

with open("../listOfMultiSenseWords_mt.txt", "r", encoding="utf-8") as multisense_mt:
    
    mt_multisense_words = multisense_mt.readlines()
    for i, word in enumerate(mt_multisense_words):
        mt_multisense_words[i] = word.replace("\n", "")
    print(mt_multisense_words)
    
    length_sentences_mt = len(mt_examples)*0.0015    
    print(length_sentences_mt)
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

# %% [markdown]
# Loop the whole target corpus to get the embeddings of the closest words on the target side

# %%
en_embeddings = []
en_words = []

length_sentences_en = math.ceil(len(en_examples["train"])*0.0015)

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

# %% [markdown]
#  Calculate cosine similarity of source multisense word and each target word in each sentence just retrieved above, 
# to see which is closest

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
#import dill
#dill.dump_session('notebook_env.db')



# %%
#import dill
#dill.load_session('notebook_env.db', errors="ignore")

# %%
results_mt = []
print(len(found_multisense_mt_info))
for i, mt_multi in enumerate(found_multisense_mt_info):
    if( i % 10 == 0):
        print(i)
    list_of_sim = []
    list_of_sim_info = []
    dist, results = model_nn.kneighbors([mt_multi['embedding'].cpu().reshape(1, -1).numpy()[0]])
    for res in results[0]:
        results_mt.append({'multi_sense_word': mt_multi['word'], 'multi_sense_sentence': mt_multi['sentence'], 'target_word': embeddings_info_to_be_mapped[res]['target_word'], 'target_sentence':  embeddings_info_to_be_mapped[res]['target_sentence']})
        
#     for i, sentence_embedding in enumerate(en_embeddings):
#         embeddings_en = [e for e in sentence_embedding]
#         print(i)
#         for j, word_embedding in enumerate(embeddings_en):
            
#             res = cosine_similarity(word_embedding.cpu().reshape(1, -1) , mt_multi['embedding'].cpu().reshape(1, -1))
#             list_of_sim.append(res)
#             list_of_sim_info.append({'multi_sense_word': mt_multi['word'], 'multi_sense_sentence': mt_multi['sentence'], 'target_word': en_words[i][j], 'target_sentence': en_words[i]})
#     list_of_biggest_sim_indices = sorted(range(len(list_of_sim)), key=lambda i: list_of_sim[i])[-5:]
#     for index in list_of_biggest_sim_indices:
#         results_mt.append(list_of_sim_info[index])

print(results_mt)

# %%
import json
with open("results_mt.txt", "w+", encoding="utf-8") as results_mt_file:
    json.dump(results_mt, results_mt_file)
    #results_mt_file.write("\n".join(json.stringify(results_mt)))

# %%
