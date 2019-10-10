import torch

from .movie_reviews import MovieReview
from .subjectivity import Subjectivity


DATASETS = ('MR', 'Subj')


def get_datasets(dataset_type, dataset_folder):
    if dataset_type not in DATASETS:
        raise ValueError("Invalid dataset: %s" % dataset_type)

    train_datasets = []
    val_datasets = []

    if dataset_type == 'MR':
        for val_cv in range(MovieReview.cv):
            train_datasets.append(MovieReview(dataset_folder, val_cv, is_val=False))
            val_datasets.append(MovieReview(dataset_folder, val_cv, is_val=True))
    elif dataset_type == 'Subj':
        for val_cv in range(Subjectivity.cv):
            train_datasets.append(Subjectivity(dataset_folder, val_cv, is_val=False))
            val_datasets.append(Subjectivity(dataset_folder, val_cv, is_val=True))

    return train_datasets, val_datasets


def sentence_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return data, target
