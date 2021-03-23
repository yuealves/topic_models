import os

import numpy as np
from tqdm import tqdm

from topic_models.utils.dictionary import Dictionary
from topic_models.utils.settings import (
    alpha, beta, K, demo_dataset_dir, iter_max, dict_file as default_dict)
from topic_models.sample import _sample


class DMM:
    """A collapsed gibbs-sampling based implementation of DMM model."""

    def __init__(self, K=K, alpha=alpha, beta=beta, n_early_stop=10,
                 iter_max=iter_max, dict_file=None, use_default_dict=False,
                 dictionary=None):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_early_stop = n_early_stop
        self.iter_max = iter_max
        self.dict_file = dict_file
        self.dictionary = dictionary
        if use_default_dict:
            self.dict_file = default_dict
    
    def _load_data(self, n_mt=None, content_list=None, tokenid_list=None):
        """You can pass data by either doc-token count matrix n_mt or a
        sequence of list consisting of words content_list(preferred)."""
        if self.dictionary is not None:
            pass
        elif self.dict_file is not None:
            self.dictionary = Dictionary(dict_file=self.dict_file)
        elif content_list is None:
            raise Exception("You should pass a dictionary file as you"
                            " don't create a dictionary with a corpus")

        if n_mt is not None:
            self.M, self.T = n_mt.shape
            self.N = n_mt.sum()
            self.w_mi = [n2s(n_m) for n_m in n_mt]
            return

        self.N = 0

        if tokenid_list is not None:
            self.T = len(self.dictionary.token_list)
            self.M = len(tokenid_list)
            self.w_mi = tokenid_list
            for doc in tokenid_list:
                self.N += len(doc)
            return

        if content_list is not None:
            if self.dictionary is None:
                self.dictionary = Dictionary(corpus=content_list,
                                             min_tf=min_tf,
                                             min_df=min_df,
                                             max_dict_len=max_dict_len,
                                             stem=stem)
            self.T = len(self.dictionary.token_list)
            self.M = len(content_list)
            self.w_mi = [[] for x in range(self.M)]
            pbar = tqdm(range(self.M), desc="Loading docs")
            for m in pbar:
                self.w_mi[m] = self.dictionary.doc2tokens(content_list[m])
                self.w_mi[m] = np.array(self.w_mi[m], dtype=np.int32)
                self.N += len(self.w_mi[m])

    def _initialize(self):
        """After data is loaded, we have to assign random topic token for each
        doc. Then we prepare some of the neccesary statistics"""

        # the topic token of the m-th doc
        self.z_m = np.random.randint(self.K, size=self.M).astype=np.int32)
        # the count of word token t in k-th topic
        self.n_kt = np.zeros((self.K, self.T), dtype=np.int32)
        for m in tqdm(range(self.M), desc="Initializing model"):
            _ = len(self.w_mi[m])
            z = self.z_m[m]
            for i in range(_):
                self.n_kt[z, self.w_mi[m][i]] += 1
        self._list2array()

    def _list2array(self):
        """Convert neccesary statistic variables to numpy one-dimensional
         array, with the purpose of following cython acceleration """
        self.W = np.zeros(self.N, dtype=np.int32)
        self.N_m = np.zeros(self.M, dtype=np.int32)
        self.I_m = np.zeros(self.M, dtype=np.int32)
        n1 = 0
        for m in range(self.M):
            self.N_m[m] = len(self.w_mi[m])
            n2 = n1 + self.N_m[m]
            self.W[n1:n2] = self.w_mi[m]
            self.I_m[m] = n1
            n1 = n2
        self.n_kt_sum = np.sum(self.n_kt, axis=1, dtype=np.int32)
