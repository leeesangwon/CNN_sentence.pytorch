import argparse
import os
import pickle

import torch
from torch.optim.adadelta import Adadelta
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from data import DATASETS, get_datasets, sentence_collate_fn
from pretrained_word2vec import PretrainedWord2Vec
from models import MODELS, get_model
import constants as const
import utils

plotter = None


def get_arguments():
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--exp_name', type=str, default='')

    # dataset
    parser.add_argument('--dataset', type=str, choices=DATASETS, default='MR')
    parser.add_argument('--dataset_folder', type=str, default='../resource/data/MR')

    # pre-trained word2vec
    parser.add_argument('--w2v_file', type=str, default='../resource/GoogleNews-vectors-negative300.bin')

    # model
    parser.add_argument('--model', type=str, choices=MODELS, default='static')

    # training
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=25)

    # outputs
    parser.add_argument('--output_root', type=str, default='../ckpt')

    # visdom
    parser.add_argument('--use_visdom', type=bool, default=True)
    parser.add_argument('--logging_root', type=str, default='../logs')

    return parser.parse_args()


def main():
    args = get_arguments()

    # expriment name
    if not args.exp_name:
        args.exp_name = '_'.join([args.dataset, args.model])
    print("# Experiment: ", args.exp_name)

    # output folder
    output_folder = os.path.join(args.output_root, args.dataset, args.exp_name)
    os.makedirs(output_folder, exist_ok=True)
    print("# Output path: ", output_folder)

    # visdom
    global plotter
    if args.use_visdom:
        logging_folder = os.path.join(args.logging_root, args.dataset, args.exp_name)
        os.makedirs(logging_folder, exist_ok=True)
        plotter = utils.VisdomLinePlotter(env_name=args.exp_name, logging_path=os.path.join(logging_folder, 'vis.log'))
        print("# Visdom path: ", logging_folder)

    # dataset
    print("# Load datasets")
    train_datasets, val_datasets, test_datasets = get_datasets(args.dataset, args.dataset_folder, args.batch_size)
    num_classes = train_datasets[0].num_classes
    vocab = train_datasets[0].vocab  # every datasets have same vocab

    # pre-trained word2vec
    print("# Load pre-trained word2vec")
    pretrained_word2vec_cache = os.path.join(os.path.dirname(args.w2v_file), args.dataset + '_w2v.pkl')
    if os.path.isfile(pretrained_word2vec_cache):
        with open(pretrained_word2vec_cache, 'rb') as f:
            pretrained_word2vec = pickle.load(f)
    else:
        pretrained_word2vec = PretrainedWord2Vec(vocab, args.w2v_file)
        with open(pretrained_word2vec_cache, 'wb') as f:
            pickle.dump(pretrained_word2vec, f)

    # train
    print("# Start training")
    for cv, (train_dataset, val_dataset, test_dataset) in enumerate(zip(train_datasets, val_datasets, test_datasets)):
        # fix random seed
        utils.fix_random_seed(seed=const.RANDOM_SEED)

        # model
        cnn = get_model(args.model, num_classes, pretrained_word2vec)
        if torch.cuda.is_available():
            cnn.cuda()

        # dataloader
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=sentence_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=sentence_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=sentence_collate_fn)

        # optimizer
        optim = Adadelta(cnn.parameters(), rho=0.95, eps=1e-6)

        # criterion
        criterion = CrossEntropyLoss()

        # training
        if plotter:
            plotter.set_cv(cv)
        output_path = os.path.join(output_folder, 'cv_%d_best.pkl' % cv)
        train(args.num_epochs, cnn, train_loader, optim, criterion, val_loader, output_path)

        # evaluation
        utils.load_model(output_path, cnn)
        accuracy = eval(cnn, test_loader)
        print('cross_val:', cv, '\taccuracy:', accuracy)


def train(num_epochs, model, dataloader, optim, criterion, val_loader, output_path):
    logger = utils.TrainLogger()
    best_val_accuracy = 0
    for ep in range(num_epochs):
        model.train()
        for i, (sentences, labels) in enumerate(dataloader):
            if utils.is_cuda(model):
                labels = labels.cuda()
            optim.zero_grad()
            preds = model(sentences)
            loss = criterion(preds, labels)
            loss.backward()
            optim.step()

            l2_norm_constraint(model, s=3)

            logger.update_iter(
                total=labels.size(0),
                correct=(labels == preds.argmax(dim=1)).sum().item(),
                loss=loss.item()
            )

        accuracy, ep_loss = logger.get_epoch()
        val_accuracy = eval(model, val_loader)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            utils.save_model(output_path, model, optim, ep)
        if plotter:
            total_iter = (ep + 1) * len(dataloader)
            plotter.plot('loss', 'epoch', 'Loss', total_iter, ep_loss)
            plotter.plot('accuracy', 'epoch', 'Accuracy', total_iter, accuracy)
            plotter.plot('accuracy', 'val', 'Accuracy', total_iter, val_accuracy)
        else:
            print('epoch: %d\t loss: %f\t accuracy: %3.2f\t test accuracy: %3.2f'
                  % (ep, ep_loss, accuracy, val_accuracy))


def l2_norm_constraint(model, s=3):
    with torch.no_grad():
        col_norms = torch.norm(model.cnn.fcl[0].weight, p=2, dim=1, keepdim=True)
        desired_norms = col_norms.clamp(0, s)
        scale = (desired_norms / (1e-7 + col_norms))
        model.cnn.fcl[0].weight *= scale


def eval(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentences, labels in dataloader:
            if utils.is_cuda(model):
                labels = labels.cuda()
            preds = model(sentences)
            total += labels.size(0)
            correct += (labels == preds.argmax(dim=1)).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    main()
