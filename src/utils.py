import numpy as np
import torch
from visdom import Visdom


class VisdomLinePlotter(object):
    def __init__(self, env_name='main', logging_path=None):
        self.viz = Visdom(log_to_filename=logging_path)
        self.env = env_name
        self.postfix = ''
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        title_name = '_'.join([title_name, self.postfix])
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel='iterations',
                    ylabel=var_name,
                )
            )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]),
                          env=self.env, win=self.plots[title_name], name=split_name, update='append')

    def set_cv(self, cv):
        self.postfix = '_'.join(['cv', str(cv)])


class TrainLogger(object):
    def __init__(self):
        self._init_iter()
        self._init_epoch()

    def _init_iter(self):
        self.iter_total = 0
        self.iter_correct = 0
        self.iter_loss = 0.0
        self.iter_count = 0

    def _init_epoch(self):
        self.ep_total = 0
        self.ep_correct = 0
        self.ep_loss = 0.0
        self.ep_count = 0

    def update_iter(self, total, correct, loss):
        self.iter_total += total
        self.iter_correct += correct
        self.iter_loss += loss
        self.iter_count += 1

    def get_iter(self):
        accuracy = 100 * self.iter_correct / self.iter_total
        self.ep_total += self.iter_total
        self.ep_correct += self.iter_correct
        loss = self.iter_loss / self.iter_count
        self.ep_loss += self.iter_loss
        self.ep_count += self.iter_count
        self._init_iter()
        return accuracy, loss

    def get_epoch(self):
        accuracy = 100 * self.ep_correct / self.ep_total
        loss = self.ep_loss / self.ep_count
        self._init_epoch()
        return accuracy, loss


def is_cuda(module):
    return next(module.parameters()).is_cuda


def fix_random_seed(seed=1905):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
