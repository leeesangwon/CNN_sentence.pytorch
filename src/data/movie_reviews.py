import os
from collections import defaultdict

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

    def __init__(self, dataset_folder, val_cv=0, is_val=False):
        super().__init__()

        self._dataset = _MovieReview(dataset_folder, self.cv)
        self.val_cv = val_cv
        self.is_val = is_val
        self.vocab = self._dataset.vocab
        self.num_classes = self._dataset.num_classes

        if self.is_val:
            self.data = self._dataset.data[val_cv]
        else:
            self.data = []
            for i, data in enumerate(self._dataset.data):
                if i == val_cv:
                    continue
                self.data.extend(data)

    def __getitem__(self, idx):
        label = self.data[idx]['y']
        sentence = self.data[idx]['text'].split()
        return sentence, label

    def __len__(self):
        return len(self.data)


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
        self.data = [[] for _ in range(self.cv)]
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
                    'num_words': len(orig_sentence.split())
                }
                self.data[split].append(datum)

    def __len__(self):
        num_data = 0
        for data in self.data:
            num_data += len(data)
        return num_data