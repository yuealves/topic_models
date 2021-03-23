from collections import defaultdict
from string import ascii_lowercase

import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm


class Dictionary:
    """Dictionary used to map word to token id or conversely."""

    def __init__(self, corpus=None, dict_file=None, token_list=None,
                 min_tf=None, min_df=None, max_dict_len=None, stem=False):
        """Pass corpus as a sequence of list consisting of words, or a
        dict_file in which every line contains a single word, or a list of
        tokens."""
        self.stem = stem
        # Create tokenizer
        self.tokenizer = RegexpTokenizer(r"[a-z]+")

        # Create English stop words list
        self.en_stop = get_stop_words('en')
        self.en_stop += list(ascii_lowercase)

        # Create p_stemmer of class PorterStemmer
        self.p_stemmer = PorterStemmer()

        if corpus is not None:
            self._from_corpus(corpus, min_tf=min_tf, min_df=min_df,
                              max_dict_len=max_dict_len, stem=stem)
        elif dict_file is not None:
            token_list = open(dict_file).read().split("\n")
            self._from_tokens_list(token_list)
        elif token_list is not None:
            self._from_tokens_list(token_list)

    def _from_corpus(self, corpus, min_tf, min_df, max_dict_len, stem):
        self._tokenTf = defaultdict(int)
        self._tokenDf = defaultdict(int)

        M = len(corpus)
        pbar = tqdm(range(M), desc="Generating dictionary from corpus")

        for m in pbar:
            doc = corpus[m]
            raw = doc.lower()
            tokens = self.tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if i not in self.en_stop]
            if self.stem:
                stemmed_tokens = [self.p_stemmer.stem(i)
                                  for i in stopped_tokens]
                words = stemmed_tokens
            else:
                words = stopped_tokens
            for word in words:
                self._tokenTf[word] += 1
            words = set(words)
            for word in words:
                self._tokenDf[word] += 1

        selected_tokens = []

        if min_tf is not None:
            self._min_tf = min_tf
            for token in self._tokenTf.keys():
                if self._tokenTf[token] >= min_tf:
                    selected_tokens.append(token)
        else:
            selected_tokens = list(self._tokenTf.keys())

        if min_df is not None:
            selected_tokens = [x for x in selected_tokens
                               if self._tokenDf[x] >= min_df]

        if max_dict_len is not None and max_dict_len < len(selected_tokens):
            _ = sorted(selected_tokens, key=lambda x: -self._tokenTf[x])
            selected_tokens = _[:max_dict_len]

        self.token_list = selected_tokens
        self._token2id = {token: id for id, token
                          in enumerate(selected_tokens)}

    def _from_tokens_list(self, token_list):
        self.token_list = token_list
        self._token2id = {token: id for id, token
                          in enumerate(self.token_list)}

    def id2token(self, tokenId):
        return self.token_list[tokenId]

    def token2id(self, tokenStr):
        if tokenStr in self._token2id:
            return self._token2id[tokenStr]
        return None

    def doc2tokens(self, doc):
        raw = doc.lower()
        tokens = self.tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if i not in self.en_stop]
        if self.stem:
            stemmed_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]
            words = stemmed_tokens
        else:
            words = stopped_tokens
        tokenids = [self.token2id(token)
                    for token in words if self.token2id(token)]
        return tokenids

    def save(self, fname="dictionary.txt"):
        with open(fname, "wt") as f:
            for token in self.token_list:
                f.write(token + "\n")
