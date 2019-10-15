import os
from collections import defaultdict
import random

import numpy as np
from torch.utils.data import Dataset

from .utils import clean_str


class MovieReview(Dataset):
    """
    Parameter:
        dataset_folder (str): a path to dataset
        val_cv (int [0, 10)): a set to use for validation within the 10 folds cross-validation.
        is_val (bool): if True, it generates dataset for validation
    Returns:
        sentence: a list of words
        label: 0 for the negative and 1 for the positive
    """
    cv = 10

    def __init__(self, dataset_folder, batch_size, test_cv=0, type='train', random_seed=1905):
        super().__init__()

        self._dataset = _MovieReview(dataset_folder, self.cv, random_seed=random_seed)
        self.test_cv = test_cv
        self.type = type.lower()
        if self.type not in ['train', 'val', 'test']:
            raise ValueError("Invalid dataset type: %s" % type)
        self.vocab = self._dataset.vocab
        self.num_classes = self._dataset.num_classes

        self._split_data(batch_size, random_seed)

    def _split_data(self, batch_size, random_seed):
        train_data, test_data = [], []
        for i, datum in enumerate(self._dataset.data):
            if datum['split'] == self.test_cv:
                test_data.append(i)
            else:
                train_data.append(i)

        random.seed(random_seed)
        if len(train_data) % batch_size > 0:
            extra_data_num = batch_size - len(train_data) % batch_size
            random.shuffle(train_data)
            extra_data = train_data[:extra_data_num]
            new_data = train_data + extra_data
        else:
            new_data = train_data
        random.shuffle(new_data)
        n_batches = len(new_data) / batch_size
        n_train_batches = int(round(n_batches * 0.9))
        train_data = new_data[:n_train_batches * batch_size]
        val_data = new_data[n_train_batches * batch_size:]

        if self.type == 'train':
            self.indexes = train_data
        elif self.type == 'val':
            self.indexes = val_data
        else:  # test
            self.indexes = test_data

    def __getitem__(self, idx):
        label = self._dataset.data[self.indexes[idx]]['y']
        sentence = self._dataset.data[self.indexes[idx]]['text'].split()
        return sentence, label

    def __len__(self):
        return len(self.indexes)


class _MovieReview(object):
    """
    Load data and split into 10 folds.
    """
    num_classes = 2
    clean_string = True

    def __init__(self, dataset_folder, cv=10, random_seed=1905):
        super().__init__()

        self.dataset_folder = dataset_folder
        self.cv = cv
        self.pos_file = os.path.join(self.dataset_folder, 'rt-polarity.pos')
        self.neg_file = os.path.join(self.dataset_folder, 'rt-polarity.neg')

        np.random.seed(random_seed)
        self.vocab = defaultdict(int)
        self.data = []
        self._read_datafile(self.pos_file, is_pos=True)
        self._read_datafile(self.neg_file, is_pos=False)

    def _read_datafile(self, file, is_pos):
        with open(file, 'rb') as f:
            for line in f:
                sentence = line.strip().decode('latin1')
                if self.clean_string:
                    orig_sentence = clean_str(sentence)
                else:
                    orig_sentence = sentence.lower()
                words = set(orig_sentence.split())
                for word in words:
                    self.vocab[word] += 1

                split = np.random.randint(0, self.cv)
                datum = {
                    'y': 1 if is_pos else 0,
                    'text': orig_sentence,
                    'num_words': len(orig_sentence.split()),
                    'split': split,
                }
                self.data.append(datum)

    def __len__(self):
        return len(self.data)
