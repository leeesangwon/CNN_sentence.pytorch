import os

import torch

import constants as const

from .cross_val_dataset import CrossValDataset, _CrossValDataset
from .sst import SSTDataset
from .trec import TRECDataset

DATASETS = ('MR', 'Subj', 'CR', 'MPQA', 'SST1', 'SST2', 'TREC')


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

    elif dataset_type in ['SST1', 'SST2']:
        datasets = []
        for dataset_filename in const.DATASET_FILENAME[dataset_type]:
            dataset_file = os.path.join(dataset_folder, dataset_filename)
            datasets.append(SSTDataset(dataset_file))

        train_datasets.append(datasets[0])
        val_datasets.append(datasets[1])
        test_datasets.append(datasets[2])

    elif dataset_type in ['TREC']:
        train_dataset_file = os.path.join(dataset_folder, const.DATASET_FILENAME[dataset_type][0])
        train_datasets = [TRECDataset(train_dataset_file, batch_size, type='train')]
        val_datasets = [TRECDataset(train_dataset_file, batch_size, type='val')]

        test_dataset_file = os.path.join(dataset_folder, const.DATASET_FILENAME[dataset_type][2])
        test_datasets = [TRECDataset(test_dataset_file, batch_size, type='test')]

    return train_datasets, val_datasets, test_datasets


def sentence_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return data, target
