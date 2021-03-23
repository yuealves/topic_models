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
