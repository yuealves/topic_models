import os

import numpy as np
from tqdm import tqdm

from topic_models.utils.dictionary import Dictionary
from topic_models.utils.settings import (
    alpha, beta, K, demo_dataset_dir, iter_max, dict_file as default_dict)
from topic_models.sample import _sample


class LDA:
    """A collapsed gibbs-sampling based implementation of LDA model."""

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

    def _load_data(self, n_mt=None, content_list=None, tokenid_list=None,
                   min_tf=None, min_df=None, max_dict_len=None, stem=True):
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
            # for m, doc in enumerate(content_list):
            #     if m % 20 == 0:
            #         print("read doc %d/%d" % (m, self.M))
            #     self.w_mi[m] = self.dictionary.doc2tokens(doc)
            #     self.w_mi[m] = np.array(self.w_mi[m], dtype=np.int32)
            #     self.N += len(self.w_mi[m])

    def _initialize(self):
        """After data is loaded, we have to assign random topic token for each
        word in the doc. Then we prepare some of the neccesary statistics"""

        # the topic token of i-th word in m-th doc
        self.z_mi = [[] for m in range(self.M)]
        # the count of word token t in k-th topic
        self.n_kt = np.zeros((self.K, self.T), dtype=np.int32)
        # the count of words assigned to k-th topic in m-th doc
        self.n_mk = np.zeros((self.M, self.K), dtype=np.int32)
        for m in tqdm(range(self.M), desc="Initializing model"):
            _ = len(self.w_mi[m])
            self.z_mi[m] = np.random.randint(self.K, size=_).astype(np.int32)
            for i, z in enumerate(self.z_mi[m]):
                self.n_mk[m, z] += 1
                self.n_kt[z, self.w_mi[m][i]] += 1
        self._list2array()

    def _list2array(self):
        """Convert neccesary statistic variables to numpy one-dimensional
         array, with the purpose of following cython acceleration """
        self.W = np.zeros(self.N, dtype=np.int32)
        self.Z = np.zeros(self.N, dtype=np.int32)
        self.N_m = np.zeros(self.M, dtype=np.int32)
        self.I_m = np.zeros(self.M, dtype=np.int32)
        n1 = 0
        for m in range(self.M):
            self.N_m[m] = len(self.w_mi[m])
            n2 = n1 + self.N_m[m]
            self.W[n1:n2] = self.w_mi[m]
            self.Z[n1:n2] = self.z_mi[m]
            self.I_m[m] = n1
            n1 = n2
        self.n_kt_sum = np.sum(self.n_kt, axis=1, dtype=np.int32)

    def _phi(self):
        """return the infered topics 'phi'. """
        smoothed = self.n_kt + self.beta
        phi = smoothed / np.c_[smoothed.sum(axis=1)]
        return phi

    def fit(self, input_data, min_tf=10, min_df=None,
            max_dict_len=None, stem=False):
        if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
            self._load_data(n_mt=input_data)
        elif isinstance(input_data[0][0], (int, np.integer)):
            self._load_data(tokenid_list=input_data)
        elif isinstance(input_data[0][0], str):
            self._load_data(content_list=input_data, min_tf=min_tf,
                            min_df=min_df, max_dict_len=max_dict_len,
                            stem=stem)
        else:
            raise Exception("Input type not supported!")

        self._initialize()

        iter_num = self.iter_max
        num_change_min = float("inf")
        num_not_min = 0
        num_z_changes = [None, ] * iter_num
        tr = tqdm(range(iter_num), desc="Training model")
        for i in tr:
            num_z_change = _sample._lda_train(self.n_mk, self.n_kt,
                                              self.n_kt_sum, self.W, self.Z,
                                              self.N_m, self.I_m,
                                              self.alpha, self.beta)
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
    lda_model = LDA(K=5, iter_max=50, n_early_stop=20)
    lda_model.fit(documents, max_dict_len=5000, min_tf=5, stem=False)
    lda_model.show_topic(top_num=15)

    # # Or use standalone dictionary
    # lda_model = LDA(K=5, n_early_stop=20, use_default_dict=True)
    # # You can also use your own dictionary
    # # lda_model = LDA(K=5, dict_file="yourdictionary.txt")
    # lda_model.fit(documents)
    # lda_model.show_topic(top_num=15)
