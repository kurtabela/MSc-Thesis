from __future__ import division
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


def get_embeddings(model, tokeniser, sentences):
    input = tokeniser.batch_encode_plus(sentences,
                                        max_length=model.embeddings.position_embeddings.num_embeddings,
                                        padding="max_length",
                                        truncation=True,
                                        # return_overflowing_tokens=True,
                                        return_tensors="pt",
                                        )
    output = model(**input)[0]

    words = []
    embeddings = []
    vocabulary = dict(zip(tokeniser.get_vocab().values(), tokeniser.get_vocab().keys()))
    for i, token in enumerate(map(lambda token_id: vocabulary[token_id],
                                  [token for instance in input["input_ids"].tolist() for token in instance])):
        try:
            embedding = output[:, i]
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

    return words, [embedding.mean(dim=0) for embedding in embeddings]


if __name__ == '__main__':
    model_name = "MLRS/BERTu"
    model, tokeniser = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)
    total_words = []
    total_embeddings = []

    with open("embeddings.tsv", "w", encoding="utf-8") as embeddings_file, \
            open("words.tsv", "w", encoding="utf-8") as words_file:
        data = datasets.load_dataset("MLRS/korpus_malti", "shuffled", split="test")
        words_file.write("word\tindex\n")
        for i, sentence in enumerate(data):
            words, embeddings = get_embeddings(model, tokeniser, [sentence["text"].strip()])
            assert len(words) == len(embeddings)

            for embedding in embeddings:
                embeddings_file.write("\t".join(str(x) for x in embedding.tolist()) + "\n")
                total_embeddings.append(embedding.tolist())
            for j, word in enumerate(words):
                words_file.write(f"{word}\t{i}_{j}\n")
                total_words.append(word)
alreadySorted = []

with open("listOfMultiSenseWords.txt", "w", encoding="utf-8") as multisensewords:
    for i, word in enumerate(total_words):
        embeddingsOfCurrentWord = []
        num_of_clusters = 2
#         print(word)


        if word in alreadySorted:
            continue
        alreadySorted.append(word)

        embeddingsOfCurrentWord.append(total_embeddings[i])

        # Cet all embeddings of the same word in diff sentences
        for j, futureWord in enumerate(total_words[i+1:]):
            if futureWord == word:
                embeddingsOfCurrentWord.append(total_embeddings[j])


        # Change the below to be in the loop so we only get the embeddings of the current word

        n_words = len(embeddingsOfCurrentWord)
        if(n_words < 2):
            continue

        eps = 19.2
        min_samples = 2
        ret = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddingsOfCurrentWord)

        if len(set(ret)) > 1:
            print(ret)
            multisensewords.write(word+"\n")
