import os
from collections import Counter

import numpy as np
from tqdm import tqdm

from topic_models.utils.dictionary import Dictionary
from topic_models.utils.func import n2s
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
                self.dictionary = Dictionary(corpus=content_list)
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
        self.z_m = np.random.randint(self.K, size=self.M).astype(np.int32)
        if self.fixed_labels is not None:
            self.z_m[:self.n_fixed] = self.fixed_labels
        # the count of word token t in k-th topic
        self.n_kt = np.zeros((self.K, self.T), dtype=np.int32)
        self.M_k = np.random.randint(self.K, size=self.M).astype(np.int32)
        for m in tqdm(range(self.M), desc="Initializing model"):
            _ = len(self.w_mi[m])
            z = self.z_m[m]
            self.M_k[z] += 1
            for i in range(_):
                self.n_kt[z, self.w_mi[m][i]] += 1
        self._list2array()

    def _list2array(self):
        """Convert neccesary statistic variables to numpy one-dimensional
         array, with the purpose of following cython acceleration """
        self.W = np.zeros(self.N, dtype=np.int32)
        self.W_freq = np.zeros(self.N, dtype=np.int32)
        self.N_m = np.zeros(self.M, dtype=np.int32)
        self.U_m = np.zeros(self.M, dtype=np.int32)
        self.I_m = np.zeros(self.M, dtype=np.int32)
        n1 = 0
        for m in range(self.M):
            self.N_m[m] = len(self.w_mi[m])
            cnt = Counter(self.w_mi[m])
            self.U_m[m] = len(cnt)
            n2 = n1 + self.U_m[m]
            self.W[n1:n2] = np.array(list(cnt.keys()))
            self.W_freq[n1:n2] = np.array(list(cnt.values()))
            self.I_m[m] = n1
            n1 = n2
        self.n_kt_sum = np.sum(self.n_kt, axis=1, dtype=np.int32)

    def fit(self, input_data, min_tf=10, min_df=None,
            max_dict_len=None, stem=False, semi_supervised_labels=None):
        if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
            self._load_data(n_mt=input_data)
        elif isinstance(input_data[0][0], (int, np.integer)):
            self._load_data(tokenid_list=input_data)
        elif isinstance(input_data[0][0], str):
            self._load_data(content_list=input_data)
        else:
            raise Exception("Input type not supported!")

        self.n_fixed = (0 if semi_supervised_labels is None
                        else len(semi_supervised_labels))
        self.fixed_labels = semi_supervised_labels
        self._initialize()

        iter_num = self.iter_max
        num_z_changes = [None, ] * iter_num
        num_change_min = float("inf")
        num_not_min = 0
        tr = tqdm(range(iter_num), desc="Training model")
        for i in tr:
            num_z_change = _sample._dmm_train(
                self.n_kt, self.n_kt_sum, self.M_k, self.I_m, self.W,
                self.W_freq, self.z_m, self.N_m, self.U_m, self.alpha,
                self.beta, self.n_fixed
            )
            num_z_changes[i] = num_z_change
            tr.set_postfix({"num_z_change": num_z_change})

            if num_z_change < num_change_min:
                num_change_min = num_z_change
                num_not_min = 0
            else:
                num_not_min += 1

            if num_not_min >= self.n_early_stop:
                tr.close()
                break
        return num_z_changes[:i+1]

    def topic_top_words(self, top_num=30):
        """Return a sequence of list of top-words in each topic"""
        result = [[] for x in range(self.K)]
        for k in range(self.K):
            topWords = self.n_kt[k, :].argsort()[::-1][:top_num]
            result[k] = [self.dictionary.id2token(t) for t in topWords]
        return result

    def show_topic(self, top_num=30):
        """Return a str format representation of topics"""
        topics = self.topic_top_words(top_num)
        str_repr = "\nExtracted topics:\n"
        for topic in topics:
            current_line = " ".join(topic)
            # print(current_line)
            # print("=" * 30)
            str_repr += ("=" * 30 + "\n" + current_line + "\n")
        print(str_repr)
        return str_repr


def demo():
    filenames = [os.path.join(demo_dataset_dir, x)
                 for x in os.listdir(demo_dataset_dir)]
    documents = [open(x, encoding="utf8").read() for x in filenames]

    # Generate corpus-based dictionary
    dmm_model = DMM(K=5, iter_max=30, n_early_stop=10)
    dmm_model.fit(documents, max_dict_len=5000, stem=False)
    dmm_model.show_topic(top_num=15)
    print(filenames)
    print(dmm_model.z_m)
