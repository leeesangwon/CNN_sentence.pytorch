import torch
from torch import nn

from utils import is_cuda


class Sentence2Mat(nn.Module):
    """
    Parameters:
        pretrained_word2vec: PretrainedWord2Vec instance
        use_pretrained: bool
        freeze: bool
    Shape:
        input: a list of words
        output: (num_words, word_vec_size)
    """
    def __init__(self, pretrained_word2vec, use_pretrained=False, freeze=False):
        super().__init__()
        self.word_vec_size = pretrained_word2vec.word_vec_size
        self.word2index = pretrained_word2vec.word2index
        self.index2vec = nn.Embedding(len(self.word2index), self.word_vec_size)
        if use_pretrained:
            self.index2vec.from_pretrained(pretrained_word2vec.embeddings, freeze)
        else:
            nn.init.uniform_(self.index2vec.weight, -0.25, 0.25)

    def forward(self, sentence):
        indexes = []
        for word in sentence:
            indexes.append(self.word2index[word])
        indexes = torch.LongTensor(indexes)
        if is_cuda(self):
            indexes = indexes.cuda()
        return self.index2vec(indexes)
