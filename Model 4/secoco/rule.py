# This file generates new data with noise

import sys
import random
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
import argparse

import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--data_out', type=str)
args = parser.parse_args()
# %%
choices = ['the', ',', '.', 'of', 'and', 'to', 'in', 'a', 'is', 'that',
           'for', 'on', 'with', 'be', 'are', 'The', 'I', 'as', 'this', 'it', 'we']

if args.src == "mt":
    choices = ['il-', ',', '.', 'ta', 'u', 'lil', 'hi', 'hu', 'dak', 'fuq', 'ma', 'huwa', 'jien', 'int', 'bħal', 'dan',
               'din', 'aħna']

if args.src == "ic":
    choices = choices = [",", ".", "sér", "sig", "sín", "aðra", "aðrar", "aðrir", "alla", "allan", "allar", "allir", "allnokkra", "allnokkrar", "allnokkrir", "allnokkru", "allnokkrum", "allnokkuð", "allnokkur", "allnokkurn", "allnokkurra", "allnokkurrar", "allnokkurri", "allnokkurs", "allnokkurt", "allra", "allrar", "allri", "alls", "allt", "allur", "annað", "annan", "annar", "annarra", "annarrar", "annarri", "annars", "báða", "báðar", "báðir", "báðum", "bæði", "beggja", "ein", "eina", "einar", "einhver", "einhverja", "einhverjar", "einhverjir", "einhverju", "einhverjum", "einhvern", "einhverra", "einhverrar", "einhverri", "einhvers", "einir", "einn", "einna", "einnar", "einni", "eins", "einskis", "einu", "einum", "eitt", "eitthvað", "eitthvert", "ekkert", "enga", "engan", "engar", "engin", "enginn", "engir", "engra", "engrar", "engri", "engu", "engum", "fáein", "fáeina", "fáeinar", "fáeinir", "fáeinna", "fáeinum", "flestalla", "flestallan", "flestallar", "flestallir", "flestallra", "flestallrar", "flestallri", "flestalls", "flestallt", "flestallur", "flestöll", "flestöllu", "flestöllum", "hin", "hina", "hinar", "hinir", "hinn", "hinna", "hinnar", "hinni", "hins", "hinu", "hinum", "hitt", "hvað", "hvaða", "hver", "hverja", "hverjar", "hverjir", "hverju", "hverjum", "hvern", "hverra", "hverrar", "hverri", "hvers", "hvert", "hvílík", "hvílíka", "hvílíkan", "hvílíkar", "hvílíkir", "hvílíkra", "hvílíkrar", "hvílíkri", "hvílíks", "hvílíkt", "hvílíku", "hvílíkum", "hvílíkur", "hvor", "hvora", "hvorar", "hvorir", "hvorn", "hvorra", "hvorrar", "hvorri", "hvors", "hvort", "hvoru", "hvorug", "hvoruga", "hvorugan", "hvorugar", "hvorugir", "hvorugra", "hvorugrar", "hvorugri", "hvorugs", "hvorugt", "hvorugu", "hvorugum", "hvorugur", "hvorum", "mestalla", "mestallan", "mestallar", "mestallir", "mestallra", "mestallrar", "mestallri", "mestalls", "mestallt", "mestallur", "mestöll", "mestöllu", "mestöllum", "mín", "mína", "mínar", "mínir", "minn", "minna", "minnar", "minni", "míns", "mínu", "mínum", "mitt", "nein", "neina", "neinar", "neinir", "neinn", "neinna", "neinnar", "neinni", "neins", "neinu", "neinum", "neitt", "nokkra", "nokkrar", "nokkrir", "nokkru", "nokkrum", "nokkuð", "nokkur", "nokkurn", "nokkurra", "nokkurrar", "nokkurri", "nokkurs", "nokkurt", "öðru", "öðrum", "öll", "öllu", "öllum", "önnur", "sá", "sama", "saman", "samar", "sami", "samir", "samra", "samrar", "samri", "sams", "samt", "samur", "sérhvað", "sérhver", "sérhverja", "sérhverjar", "sérhverjir", "sérhverju", "sérhverjum", "sérhvern", "sérhverra", "sérhverrar", "sérhverri", "sérhvers", "sérhvert", "sín", "sína", "sínar", "sínhver", "sínhverja", "sínhverjar", "sínhverjir", "sínhverju", "sínhverjum", "sínhvern", "sínhverra", "sínhverrar", "sínhverri", "sínhvers", "sínhvert", "sínhvor", "sínhvora", "sínhvorar", "sínhvorir", "sínhvorn", "sínhvorra", "sínhvorrar", "sínhvorri", "sínhvors", "sínhvort", "sínhvoru", "sínhvorum", "sínir", "sinn", "sinna", "sinnar", "sinnhver", "sinnhverja", "sinnhverjar", "sinnhverjir", "sinnhverju", "sinnhverjum", "sinnhvern", "sinnhverra", "sinnhverrar", "sinnhverri", "sinnhvers", "sinnhvert", "sinnhvor", "sinnhvora", "sinnhvorar", "sinnhvorir", "sinnhvorn", "sinnhvorra", "sinnhvorrar", "sinnhvorri", "sinnhvors", "sinnhvort", "sinnhvoru", "sinnhvorum", "sinni", "síns", "sínu", "sínum", "sitt", "sitthvað", "sitthver", "sitthverja", "sitthverjar", "sitthverjir", "sitthverju", "sitthverjum", "sitthvern", "sitthverra", "sitthverrar", "sitthverri", "sitthvers", "sitthvert", "sitthvor", "sitthvora", "sitthvorar", "sitthvorir", "sitthvorn", "sitthvorra", "sitthvorrar", "sitthvorri", "sitthvors", "sitthvort", "sitthvoru", "sitthvorum", "sjálf", "sjálfa", "sjálfan", "sjálfar", "sjálfir", "sjálfra", "sjálfrar", "sjálfri", "sjálfs", "sjálft", "sjálfu", "sjálfum", "sjálfur", "slík", "slíka", "slíkan", "slíkar", "slíkir", "slíkra", "slíkrar", "slíkri", "slíks", "slíkt", "slíku", "slíkum", "slíkur", "söm", "sömu", "sömum", "sú", "sum", "suma", "suman", "sumar", "sumir", "sumra", "sumrar", "sumri", "sums", "sumt", "sumu", "sumum", "sumur", "vettugi", "vor", "vora", "vorar", "vorir", "vorn", "vorra", "vorrar", "vorri", "vors", "vort", "voru", "vorum", "ýmis", "ýmiss", "ýmissa", "ýmissar", "ýmissi", "ýmist", "ýmsa", "ýmsan", "ýmsar", "ýmsir", "ýmsu", "ýmsum", "þá", "það", "þær", "þann", "þau", "þeim", "þeir", "þeirra", "þeirrar", "þeirri", "þennan", "þess", "þessa", "þessar", "þessara", "þessarar", "þessari", "þessi", "þessir", "þessu", "þessum", "þetta", "þín", "þína", "þínar", "þínir", "þinn", "þinna", "þinnar", "þinni", "þíns", "þínu", "þínum", "þitt", "þónokkra", "þónokkrar", "þónokkrir", "þónokkru", "þónokkrum", "þónokkuð", "þónokkur", "þónokkurn", "þónokkurra", "þónokkurrar", "þónokkurri", "þónokkurs", "þónokkurt", "því", "þvílík", "þvílíka", "þvílíkan", "þvílíkar", "þvílíkir", "þvílíkra", "þvílíkrar", "þvílíkri", "þvílíks", "þvílíkt", "þvílíku", "þvílíkum", "þvílíkur", "að", "af", "alltað", "andspænis", "auk", "austan", "austanundir", "austur", "á", "án", "árla", "ásamt", "bak", "eftir", "fjarri", "fjær", "fram", "frá", "fyrir", "gagnstætt", "gagnvart", "gegn", "gegnt", "gegnum", "handa", "handan", "hjá", "inn", "innan", "innanundir", "í", "jafnframt", "jafnhliða", "kring", "kringum", "með", "meðal", "meður", "miðli", "milli", "millum", "mót", "móti", "nálægt", "neðan", "niður", "norðan", "nær", "nærri", "næst", "næstum", "of", "ofan", "ofar", "óháð", "órafjarri", "sakir", "samfara", "samhliða", "samkvæmt", "samskipa", "samtímis", "síðan", "síðla", "snemma", "sunnan", "sökum", "til", "tráss", "um", "umfram", "umhverfis", "undan", "undir", "upp", "utan", "úr", "út", "útundan", "vegna", "vestan", "vestur", "við", "viður", "yfir", "hið", "hin", "hina", "hinar", "hinir", "hinn", "hinna", "hinnar", "hinni", "hins", "hinu", "hinum", "ég", "hana", "hann", "hans", "hennar", "henni", "honum", "hún", "mér", "mig", "mín", "okkar", "okkur", "oss", "vér", "við", "vor", "yðar", "yður", "ykkar", "ykkur", "þá", "það", "þær", "þau", "þeim", "þeir", "þeirra", "þér", "þess", "þið", "þig", "þín", "þú", "því", "að", "annaðhvort", "bæði", "eða", "eður", "ef", "eftir", "ella", "ellegar", "en", "enda", "er", "fyrst", "heldur", "hvenær", "hvorki", "hvort", "meðan", "nema", "né", "nú", "nær", "og", "sem", "síðan", "svo", "til", "um", "uns", "utan", "ýmist", "þar", "þá", "þegar", "þó", "þótt", "því"]



for split in ['train', 'dev', 'test']:
    with open(args.data_out + split + '.bpe.' + args.src, 'r') as src, open(args.data_out + split + '.bpe.tag.' + args.src,
                                                                        'w+') as w:
        
        src_tokens = [token for token in src.read().split() if len(token.strip()) > 0]
        
        # pro
        temp = []
        prev_bpe = False
        present_bpe = False
        print(len(src_tokens))
        for i, token in enumerate(src_tokens):
            #print(i)
            # ignore bpe words
            if token.endswith('@@'):
                present_bpe = True
            else:
                present_bpe = False

            # random delete
            if random.random() < 0.02 and not present_bpe and not prev_bpe and '-' not in token:
                symbol = f'<Delete-[{token}]>'
                temp.append(symbol)
            # random insert
            elif random.random() < 0.02 and not present_bpe and not prev_bpe:
                # repeat 1-3 times
                if random.random() <= 0.7 and '-' not in token:
                    temp.append(token)
                    t = random.randint(1, 3)
                    t_tokens = [token] * t
                    symbol = f"<InsertRepeat-[{'|'.join(t_tokens)}]>"
                # random insert
                else:
                    temp.append(token)
                    symbol = f'<InsertRandom-[{random.choice(choices)}]>'
                temp.append(symbol)
            else:
                temp.append(token)

            if token.endswith('@@'):
                prev_bpe = True
            else:
                prev_bpe = False

        w.write(f"{' '.join(temp)}\n")
