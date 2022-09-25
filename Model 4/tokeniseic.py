#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.tokenize import RegexpTokenizer
import sentence_splitter
import sentencepiece as spm
from abc import ABC, abstractmethod
import re
import os
import argparse
import tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', '--src', type=str, help='File with each line-by-line model outputs')
    parser.add_argument('--target', '--tgt', type=str, help='Output file name')
    parser.add_argument('--encode', action='store_const', const=True, default=False)
    parser.add_argument('--decode', action='store_const', const=True, default=False)
    args = parser.parse_args()

    if args.encode:
        system_lines = []
        with open(args.source, 'r') as f:
            for line in f:
                system_lines.append(line.strip())
        with open(args.target, 'w+') as w:
            print(args.target)
            for line in system_lines:
                txt = tokenizer.tokenize((line))
                txt = list(map((lambda x: x[1]), txt))
                w.write(' '.join(txt).strip() + '\n')


    if args.decode:
        system_lines = []
        with open(args.source, 'r') as f:
            for line in f:
                system_lines.append(line.strip())
        with open(args.target, 'w+') as w:
            for line in system_lines:
                toklist = list(tokenizer.tokenize(line.split()))
                w.write(tokenizer.detokenize(toklist, normalize=True))
                w.write('\n')

