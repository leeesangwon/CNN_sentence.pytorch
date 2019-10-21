import os

import torch

import constants as const

from .cross_val_dataset import CrossValDataset, _CrossValDataset

DATASETS = ('MR', 'Subj', 'CR', 'MPQA')


def get_datasets(dataset_type, dataset_folder, batch_size):
    if dataset_type not in DATASETS:
        raise ValueError("Invalid dataset: %s" % dataset_type)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    if dataset_type in ['MR', 'Subj', 'CR', 'MPQA']:
        dataset_file = os.path.join(dataset_folder, const.DATASET_FILENAME[dataset_type])
        dataset = _CrossValDataset(dataset_file, CrossValDataset.cv, random_seed=const.RANDOM_SEED)
        for test_cv in range(CrossValDataset.cv):
            train_datasets.append(CrossValDataset(dataset, batch_size, test_cv, type='train'))
            val_datasets.append(CrossValDataset(dataset, batch_size, test_cv, type='val'))
            test_datasets.append(CrossValDataset(dataset, batch_size, test_cv, type='test'))

    return train_datasets, val_datasets, test_datasets


def sentence_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return data, target
