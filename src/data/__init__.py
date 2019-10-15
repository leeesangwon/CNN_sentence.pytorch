import torch

from .movie_reviews import MovieReview
from .subjectivity import Subjectivity


DATASETS = ('MR', 'Subj')


def get_datasets(dataset_type, dataset_folder, batch_size):
    if dataset_type not in DATASETS:
        raise ValueError("Invalid dataset: %s" % dataset_type)

    train_datasets = []
    val_datasets = []
    test_datasets = []

    if dataset_type == 'MR':
        for test_cv in range(MovieReview.cv):
            train_datasets.append(MovieReview(dataset_folder, batch_size, test_cv, type='train'))
            val_datasets.append(MovieReview(dataset_folder, batch_size, test_cv, type='val'))
            test_datasets.append(MovieReview(dataset_folder, batch_size, test_cv, type='test'))
    elif dataset_type == 'Subj':
        for test_cv in range(Subjectivity.cv):
            train_datasets.append(Subjectivity(dataset_folder, batch_size, test_cv, type='train'))
            val_datasets.append(Subjectivity(dataset_folder, batch_size, test_cv, type='val'))
            test_datasets.append(Subjectivity(dataset_folder, batch_size, test_cv, type='test'))

    return train_datasets, val_datasets, test_datasets


def sentence_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return data, target
