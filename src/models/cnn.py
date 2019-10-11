# TODO: Create L2 constraint

import torch
from torch import nn

from utils import is_cuda
from .sentence2matrix import Sentence2Mat


class CNN(nn.Module):
    """
    Parameters:
        num_classes: int
        pretrained_word2vec: PretraindWord2Vec instance
        use_pretrained: bool
        freeze: bool
    Shape:
        input: batch * list of words
        outputs: (batch, num_classes)
    """
    def __init__(self, num_classes, pretrained_word2vec, use_pretrained, freeze):
        super().__init__()
        self.sentence2mat = Sentence2Mat(pretrained_word2vec, use_pretrained=use_pretrained, freeze=freeze)

        self.word_vec_size = pretrained_word2vec.word_vec_size
        self.cnn = _CNN(self.word_vec_size, num_classes)

    def forward(self, input_):
        sentences = []
        max_len = 0
        for words in input_:
            if len(words) > max_len:
                max_len = len(words)
            sentence_matrix = self.sentence2mat(words)
            sentences.append(sentence_matrix)

        # zero padding
        max_len = max(max_len, 5)
        batch_size = len(input_)
        sentence_tensor = torch.zeros(batch_size, max_len, self.word_vec_size)
        for i, sentence_matrix in enumerate(sentences):
            sentence_tensor[i].narrow(0, 0, sentence_matrix.size(0)).copy_(sentence_matrix)

        if is_cuda(self):
            sentence_tensor = sentence_tensor.cuda()

        x = self.cnn(sentence_tensor)
        return x


class CNNMultiChannel(nn.Module):
    """
    Parameters:
        num_classes: int
        pretrained_word2vec: PretrainedWord2Vec instance
    Shape:
        input: batch * list of words
        outputs: (batch, num_classes)
    """
    def __init__(self, num_classes, pretrained_word2vec):
        super().__init__()
        self.sentence2mat_static = Sentence2Mat(pretrained_word2vec, use_pretrained=True, freeze=True)
        self.sentence2mat_non_static = Sentence2Mat(pretrained_word2vec, use_pretrained=True, freeze=False)

        self.word_vec_size = pretrained_word2vec.word_vec_size
        self.cnn = _CNNMultiChannel(self.word_vec_size, num_classes)

    def forward(self, input_):
        sentences = []
        max_len = 0
        for words in input_:
            if len(words) > max_len:
                max_len = len(words)
            sentence_matrix_static = self.sentence2mat_static(words)
            sentence_matrix_non_static = self.sentence2mat_non_static(words)
            sentence_matrix = torch.stack((sentence_matrix_static, sentence_matrix_non_static), dim=-1)
            sentences.append(sentence_matrix)

        # zero padding
        max_len = max(max_len, 5)
        batch_size = len(input_)
        sentence_tensor = torch.zeros(batch_size, max_len, self.word_vec_size, 2)
        for i, sentence_matrix in enumerate(sentences):
            sentence_tensor[i].narrow(0, 0, sentence_matrix.size(0)).copy_(sentence_matrix)

        x = self.cnn(sentence_tensor)
        return x


class _CNN(nn.Module):
    """
    Parameters:
        word_vector_size: int
        out_features: int
        filter_windows: tuple of int
        num_filter: int
        drop_rate: float [0, 1]
    Shape:
        input: (batch, num_word, word_vector_size)
        output: (batch, out_features)
    """
    def __init__(self, word_vector_size=300, out_features=2, filter_windows=(3, 4, 5), num_filter=100, drop_rate=0.5):
        super().__init__()
        convs = []
        for h in filter_windows:
            conv = self._build_conv(word_vector_size, num_filter, h)
            convs.append(
               nn.Sequential(
                    conv,
                    nn.ReLU(inplace=True),
                )
            )
        self.convs = nn.ModuleList(convs)
        self.max_over_time = nn.AdaptiveMaxPool1d(1)
        fcl = self._build_fcl(num_filter*len(filter_windows), out_features)
        self.fcl = nn.Sequential(
            fcl,
            nn.Dropout(p=drop_rate),
        )

    @staticmethod
    def _build_conv(in_channels, out_channels, kernel_size, non_linear='relu'):
        conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        if non_linear in ['none', 'relu']:
            nn.init.uniform_(conv.weight, -0.01, 0.01)
        else:
            nn.init.xavier_normal_(conv.weight, gain=1.0)
        nn.init.zeros_(conv.bias)
        return conv

    @staticmethod
    def _build_fcl(in_channels, out_channels, non_linear='relu'):
        fcl = nn.Linear(in_channels, out_channels)
        if non_linear in ['none', 'relu']:
            nn.init.normal_(fcl.weight, 0, 0.01)
        else:
            nn.init.xavier_normal_(fcl.weight, gain=1.0)
        nn.init.zeros_(fcl.bias)
        return fcl

    def forward(self, x):
        x = x.transpose(1, 2)         # (batch, word_vec_size, num_word)
        x = self._conv_and_pool(x)    # (batch, num_filters*len(filter_windows), 1)
        x = x.transpose(1, 2)         # (batch, 1, num_filters*len(filter_windows))
        x = self.fcl(x)               # (batch, 1, out_features)
        return x.squeeze(1)           # (batch, out_features)

    def _conv_and_pool(self, x):
        """
        Shape:
            input: (batch, word_vec_size, num_word)
            output: (batch, num_filters*len(filter_windows), 1)
        """
        after_pooling = []
        for conv in self.convs:
            xx = conv(x)                 # (batch, num_filters, num_word-h+1)
            xx = self.max_over_time(xx)  # (batch, num_filters, 1)
            after_pooling.append(xx)

        return torch.cat(after_pooling, dim=1)  # (batch, num_filters*len(filter_windows), 1)


class _CNNMultiChannel(_CNN):
    """
    Parameters:
        word_vector_size: int
        out_features: int
        filter_windows: tuple of int
        num_filter: int
        drop_rate: float [0, 1]
    Shape:
        input: (batch, num_word, word_vector_size, 2)
        output: (batch, out_features)
    """

    def __init__(self, word_vector_size=300, out_features=2, filter_windows=(3, 4, 5), num_filter=100, drop_rate=0.5):
        super().__init__(word_vector_size, out_features, filter_windows, num_filter, drop_rate)

    def _conv_and_pool(self, x):
        """
        Shape:
            input: (batch, word_vec_size, num_word, 2)
            output: (batch, num_filters*len(filter_windows), 1)
        """
        batch_size = x.shape[0]
        after_polling = []
        for conv in self.convs:
            xx = torch.cat((x[:, :, :, 0], x[:, :, :, 1]), dim=0)  # (batch*2, word_vec_size, num_word)
            xx = conv(xx)                                          # (batch*2, num_filters, num_word-h+1)
            xx = xx[:batch_size] + xx[batch_size:]                 # (batch, num_filters, num_word-h+1)
            xx = self.max_over_time(xx)                            # (batch, num_filters, 1)
            after_polling.append(xx)

        return torch.cat(after_polling, dim=1)  # (batch, num_filters*len(filter_windows), 1)
