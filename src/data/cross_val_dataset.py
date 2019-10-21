import random

from torch.utils.data import Dataset
import numpy as np

import constants as const
from utils import fix_random_seed
from .base_dataset import _BaseDataset


class CrossValDataset(Dataset):
    cv = 10

    def __init__(self, dataset, batch_size, test_cv=0, type='train', random_seed=const.RANDOM_SEED):
        """

        :param dataset: a _CrossValDataset instance
        :param batch_size: int
        :param test_cv: cv for the test
        :param type: str, `train`, `val` or `test`
        :param random_seed:
        """
        super().__init__()

        self._dataset = dataset
        self.type = type.lower()
        if self.type not in ['train', 'val', 'test']:
            raise ValueError("Invalid dataset type: %s" % type)
        self.vocab = self._dataset.vocab
        self.num_classes = self._dataset.num_classes

        train_indexes, test_indexes = self._split_test_data(test_cv)
        train_indexes, dev_indexes = self._split_dev_data(train_indexes, batch_size, random_seed)
        if self.type == 'train':
            self.indexes = train_indexes
        elif self.type == 'val':
            self.indexes = dev_indexes
        else:  # test
            self.indexes = test_indexes

    def _split_test_data(self, test_cv):
        train_indexes, test_indexes = [], []
        for i, datum in enumerate(self._dataset.data):
            if datum['split'] == test_cv:
                test_indexes.append(i)
            else:
                train_indexes.append(i)
        return train_indexes, test_indexes

    @staticmethod
    def _split_dev_data(train_indexes, batch_size, random_seed):
        random.seed(random_seed)
        if len(train_indexes) % batch_size > 0:
            extra_data_num = batch_size - len(train_indexes) % batch_size
            random.shuffle(train_indexes)
            extra_data = train_indexes[:extra_data_num]
            new_data = train_indexes + extra_data
        else:
            new_data = train_indexes
        random.shuffle(new_data)
        n_batches = len(new_data) / batch_size
        n_train_batches = int(round(n_batches * 0.9))
        train_indexes = new_data[:n_train_batches * batch_size]
        dev_indexes = new_data[n_train_batches * batch_size:]

        return train_indexes, dev_indexes

    def __getitem__(self, idx):
        label = self._dataset.data[self.indexes[idx]]['y']
        sentence = self._dataset.data[self.indexes[idx]]['text'].split()
        return sentence, label

    def __len__(self):
        return len(self.indexes)


class _CrossValDataset(_BaseDataset):
    """
    Load data and split into 10 folds.
    """
    def __init__(self, dataset_file, cv=10, random_seed=const.RANDOM_SEED):
        super().__init__(dataset_file)
        if cv <= 0:
            raise ValueError("num_cv should be larger than 10, but %d" % cv)
        self.cv = cv
        fix_random_seed(random_seed)
        self._split_cross_val_data()

    def _split_cross_val_data(self):
        for datum in self.data:
            split = np.random.randint(0, self.cv)
            datum['split'] = split
