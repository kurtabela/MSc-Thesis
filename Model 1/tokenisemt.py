#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:28:27 2017

@author: albert
"""
from nltk.tokenize import RegexpTokenizer
import sentence_splitter
import sentencepiece as spm
from abc import ABC, abstractmethod
import re
import os
import argparse

class MTRegex:
    # Definite article + prepositions with cliticised article
    # DEF_ARTICLE = r'da|di|b[ħh]a|g[hħ]a|b[ħh]al|g[ħh]al|lil?|sa|ta|ma|fi?|mil?|bil?|[ġg]o|i|sa|bi[dtlrnsxzcżċ]-'
    DEF_ARTICLE = r'\w{0,5}?[dtlrnsxzcżċ]-'
    DEF_NUMERAL = r'-i[dtlrnsxzcżċ]'

    L_APOST = r"[’'](i?)l-?";

    # Apostrophised prepositions
    APOST = r'\w+a|[mtxbfs][\'’]'

    NUMBER = r'\d+'
    DECIMAL = r'\d+[\.,/]\d+'

    # All other tokens: string of alphanumeric chars, numbers or a single
    # non-alphanumeric char. (Accent or apostrophe allowed at end of string of alpha chars).
    WORD = r'\w+[`\']?|\S'

    ALPHA_WORD = "\w+";

    ALL_WORDS = DEF_ARTICLE + "|" + DEF_NUMERAL + "|" + L_APOST + "|" + APOST + "|" + WORD + "|" + ALPHA_WORD;

    END_PUNCTUATION = r'\?|\.|,|\!|;|:|…|"|\'|\.\.\.\''

    PROCLITIC_PREP = r"^\w['’]$"

    ABBREV_PREFIX = r"sant['’]|(a\.?m|p\.?m|onor|sra|nru|dott|kap|mons|dr|prof)\.?"

    NUMERIC_DATE = r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}"

    # Courtesy of https://www.geeksforgeeks.org/python-check-url-string/
    URL = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    # URL = "(((http|ftp|gopher|javascript|telnet|file|ssh|scp)://(www\.)?)|mailto:|www\.).+\\s$";
    # URL2 = "(((http|ftp|https|gopher|javascript|telnet|file)://)|(www\.)|(mailto:))[\w\-_]+(\.[\w\-_]+)?([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?";

    # Courtesy of https://stackoverflow.com/questions/201323/how-to-validate-an-email-address-using-a-regular-expression
    EMAIL = r'''(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'''

    # It-Tlieta, 25 ta' Frar, 2003
    FULL_DATE = "(^I[ltnsdxr]-(Tnejn|Tlieta|Erbgħa|Ħamis|Ġimgħa|Sibt|Ħadd), \d{1,2} ta' " + "(Jannar|Frar|Marzu|April|Mejju|Ġunju|Lulju|Awwissu|Settembru|Ottubru|Novembru|Diċembru), \d{4,}$)"

    # Hyphen-separated prefixes which shouldn't be separated from following words
    PREFIX = "(sotto|inter|intra|mini|ex|eks|pre|post|sub|neo|soċjo)-";

    # All tokens: definite article or token
    # TOKEN = DEF_ARTICLE + "|" + DEF_NUMERAL + "|" + APOST + "|" + L_APOST + "|" + ABBREV_PREFIX + "|" + NUMERIC_DATE + "|" + NUMBER + "|" + WORD;
    TOKEN = DEF_ARTICLE + "|" + DEF_NUMERAL + "|" + WORD + "|" + END_PUNCTUATION

    ALPHA_TOKEN = DEF_ARTICLE + "|" + DEF_NUMERAL + "|" + APOST + "|" + L_APOST + "|" + ABBREV_PREFIX + "|" + ALPHA_WORD;

    BLANK_LINE = r"^\s*[\r\n]+\s*$"

    @staticmethod
    def is_word(string):
        return re.match(MTRegex.ALPHA_TOKEN, string) is not None

    @staticmethod
    def is_end_of_sentence(word, next):
        if next is None:
            return False
        else:
            return MTRegex.is_end_punctuation(next) and not MTRegex.is_suffix(next) and not MTRegex.is_prefix(word)

    @staticmethod
    def is_end_punctuation(string):
        return re.match(MTRegex.END_PUNCTUATION, string) is not None

    @staticmethod
    def is_prefix(string):
        pattern = re.compile(MTRegex.DEF_ARTICLE + "|" + MTRegex.PROCLITIC_PREP + "|" + MTRegex.ABBREV_PREFIX,
                             re.IGNORECASE)
        return re.match(pattern, string) is not None

    @staticmethod
    def is_suffix(string):
        pattern = re.compile(MTRegex.DEF_NUMERAL, re.IGNORECASE)
        return re.match(pattern, string) is not None

    @staticmethod
    def is_url(string):
        pattern = re.compile(MTRegex.URL)
        return re.match(pattern, string) is not None

    @staticmethod
    def is_email_address(string):
        return re.match(MTRegex.EMAIL, string) is not None


class MTTokenizer(ABC):

    @abstractmethod
    def tokenize(self, text):
        pass


class MTWordTokenizer(RegexpTokenizer, MTTokenizer):

    def __init__(self):
        '''Initialise an MTTokenizer with a sequence of regexps that match different tokens.
        These are internally compiled in a disjunction (a|b|..|z)'''

        self._args = [MTRegex.NUMERIC_DATE, MTRegex.DECIMAL, MTRegex.NUMBER,
                      MTRegex.DEF_ARTICLE, MTRegex.DEF_NUMERAL,
                      MTRegex.PROCLITIC_PREP, MTRegex.WORD,
                      MTRegex.END_PUNCTUATION]

        super().__init__("|".join(self._args), gaps=False, discard_empty=True,
                         flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)

    def tokenize_fix_quotes(self, text):
        text = re.sub(u'[\u201c\u201d]', '"', text)
        text = re.sub(u'[\u2018\u2019]', "'", text)
        return self.tokenize(text)

    def detokenize(self, tokens):
        text = ''

        for tok in tokens:
            if MTRegex.is_prefix(tok):
                text += tok
            elif MTRegex.is_end_punctuation(tok) or MTRegex.is_suffix(tok):
                if text.endswith(' '):
                    text = text[:-1] + f"{tok} "
            else:
                text += f"{tok} "

        return text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', '--src', type=str, help='File with each line-by-line model outputs')
    parser.add_argument('--target', '--tgt', type=str, help='Output file name')
    parser.add_argument('--encode', action='store_const', const=True, default=False)
    parser.add_argument('--decode', action='store_const', const=True, default=False)
    args = parser.parse_args()

    tok = MTWordTokenizer()

    if args.encode:
        system_lines = []
        with open(args.source, 'r') as f:
            for line in f:
                system_lines.append(line.strip())
        with open(args.target, 'w+') as w:
            print(args.target)
            for line in system_lines:
                w.write(' '.join(tok.tokenize(line)) + '\n')


    if args.decode:
        system_lines = []
        with open(args.source, 'r') as f:
            for line in f:
                system_lines.append(line.strip())
        with open(args.target, 'w+') as w:
            for line in system_lines:
                w.write(tok.detokenize(line.split()))
                w.write('\n')

