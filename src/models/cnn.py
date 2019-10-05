import torch
from torch import nn

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
        batch_size = len(input_)
        sentence_tensor = torch.zeros(batch_size, max_len, self.word_vec_size)
        for i, sentence_matrix in enumerate(sentences):
            sentence_tensor[i].narrow(0, 0, sentence_matrix.size(0)).copy_(sentence_matrix)

        x = self.cnn(sentence_tensor)
        return x


class _CNN(nn.Module):
    """
    Parameters:
        filter_windows:
        num_filter:
        word_vector_size:
        out_features:
    Shape:
        input: (batch, num_word, word_vector_size)
        output: (batch, out_features)
    """
    def __init__(self, word_vector_size=300, out_features=2, filter_windows=(3, 4, 5), num_filter=100, drop_rate=0.5):
        super().__init__()
        self.convs = []
        for h in filter_windows:
            self.convs.append(
               nn.Sequential(
                    nn.Conv1d(
                        in_channels=word_vector_size,
                        out_channels=num_filter,
                        kernel_size=h,
                    ),
                    nn.ReLU(),
                )
            )
        self.max_over_time = nn.AdaptiveMaxPool1d(1)
        self.fcl = nn.Sequential(
            nn.Linear(num_filter*len(filter_windows), out_features),
            nn.Dropout(p=drop_rate),
        )

    def forward(self, x):
        x = x.transpose(1, 2)         # (batch, word_vec_size, num_word)

        after_pooling = []
        for conv in self.convs:
            xx = conv(x)                 # (batch, num_filters, num_word-h+1)
            xx = self.max_over_time(xx)  # (batch, num_filters, 1)
            after_pooling.append(xx)

        x = torch.cat(after_pooling, dim=1)  # (batch, num_filters*len(filter_windows), 1)
        x = x.transpose(1, 2)  # (batch, 1, num_filters*len(filter_windows))
        x = self.fcl(x)        # (batch, 1, out_features)
        return x.squeeze(1)    # (batch, out_features)
