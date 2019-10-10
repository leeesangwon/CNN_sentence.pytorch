import argparse

import torch
from torch.optim.adadelta import Adadelta
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from data import DATASETS, get_datasets, sentence_collate_fn
from pretrained_word2vec import PretrainedWord2Vec
from models import MODELS, get_model


def get_arguments():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, choices=DATASETS, default='MR')
    parser.add_argument('--dataset_folder', type=str, default='../resource/MR')

    # pre-trained word2vec
    parser.add_argument('--w2v_file', type=str, default='../resource/GoogleNews-vectors-negative300.bin')

    # model
    parser.add_argument('--model', type=str, choices=MODELS, default='static')

    # training
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=25)

    return parser.parse_args()


def main():
    args = get_arguments()

    # dataset
    train_datasets, val_datasets = get_datasets(args.dataset, args.dataset_folder)
    num_classes = train_datasets[0].num_classes
    vocab = train_datasets[0].vocab  # every datasets have same vocab

    # pre-trained word2vec
    pretrained_word2vec = PretrainedWord2Vec(vocab, args.w2v_file)

    for i, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        # model
        cnn = get_model(args.model, num_classes, pretrained_word2vec)

        # dataloader
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=sentence_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=sentence_collate_fn)

        # optimizer
        optim = Adadelta(cnn.parameters(), rho=0.95, eps=1e-6)

        # criterion
        criterion = CrossEntropyLoss()

        train(args.num_epochs, cnn, train_loader, optim, criterion)
        accuracy = eval(cnn, val_loader)
        print('cross_val:', i, '\taccuracy:', accuracy)


def train(num_epochs, model, dataloader, optim, criterion):
    model.train()
    for ep in range(num_epochs):
        for i, (sentences, labels) in enumerate(dataloader):
            optim.zero_grad()
            preds = model(sentences)
            loss = criterion(preds, labels)
            loss.backward()
            optim.step()


def eval(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentences, labels in dataloader:
            preds = model(sentences)
            total += labels.size(0)
            correct += (labels == preds.argmax(dim=1)).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    main()
