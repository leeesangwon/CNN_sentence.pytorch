from torch.utils.data import Dataset

import constants as const
from utils import split_dev_data

from .base_dataset import _BaseDataset


class TRECDataset(Dataset):
    def __init__(self, dataset_file, batch_size, type='train', random_seed=const.RANDOM_SEED):
        super().__init__()
        self._dataset = _BaseDataset(dataset_file, is_trec=True)
        self.type = type
        self.vocab = self._dataset.vocab
        self.num_classes = self._dataset.num_classes

        indexes = list(range(len(self._dataset)))
        if self.type == 'test':
            self.indexes = indexes
        else:
            # randomly select 10%$ of the training set as the dev set
            train_indexes, dev_indexes = split_dev_data(indexes, batch_size, random_seed)
            if self.type == 'train':
                self.indexes = train_indexes
            elif self.type == 'val':
                self.indexes = dev_indexes

    def __getitem__(self, idx):
        label = self._dataset.data[self.indexes[idx]]['y']
        sentence = self._dataset.data[self.indexes[idx]]['text'].split()
        return sentence, label

    def __len__(self):
        return len(self.indexes)
