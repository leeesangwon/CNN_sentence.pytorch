import unittest

import torch

from ..cnn import CNN, CNNMultiChannel, _CNN, _CNNMultiChannel
from src.pretrained_word2vec import PretrainedWord2Vec
from src.pretrained_word2vec.tests.test_pretrained_word2vec import W2V_FILE


class CNNTest(unittest.TestCase):
    def test_output_shape_random(self):
        self._test_output_shape(False, False)

    def test_output_shape_static(self):
        self._test_output_shape(True, True)

    def test_output_shape_non_static(self):
        self._test_output_shape(True, False)

    def test_output_shape_num_classes(self):
        self._test_output_shape(True, False, 5)

    def _test_output_shape(self, use_pretrained, freeze, num_classes=2):
        word_list = ['head', 'body', 'hand', 'foot']
        pretrained_word2vec = PretrainedWord2Vec(word_list, W2V_FILE)
        cnn = CNN(num_classes, pretrained_word2vec, use_pretrained, freeze)
        sentences = [['head', 'body', 'hand', 'hand', 'foot', 'foot'],
                     ['body', 'head']]
        pred = cnn(sentences)
        self.assertEqual(list(pred.shape), [len(sentences), num_classes])


class CNNMultiChannelTest(unittest.TestCase):
    def test_output_shape(self):
        self._test_output_shape()
        self._test_output_shape(5)

    def _test_output_shape(self, num_classes=2):
        word_list = ['head', 'body', 'hand', 'foot']
        pretrained_word2vec = PretrainedWord2Vec(word_list, W2V_FILE)
        cnn = CNNMultiChannel(num_classes, pretrained_word2vec)
        sentences = [['head', 'body', 'hand', 'hand', 'foot', 'foot'],
                     ['body', 'head']]
        pred = cnn(sentences)
        self.assertEqual(list(pred.shape), [len(sentences), num_classes])


class _CNNTest(unittest.TestCase):
    def test_shape(self):
        b, n, k = 50, 20, 300
        out_features = 2

        x = self._define_input(b, n, k)
        cnn = self._define_cnn(out_features=out_features)
        y = cnn(x)
        self.assertEqual(list(y.shape), [b, out_features])

    def test_filter_windows(self):
        b, n, k = 50, 20, 300
        out_features = 2
        filter_windows = [
            (3, 4, 5),
            (2, 3, 4, 5),
            (7, 7, 7),
        ]

        x = self._define_input(b, n, k)
        for windows in filter_windows:
            cnn = self._define_cnn(out_features=out_features, filter_windows=windows)
            y = cnn(x)
            self.assertEqual(list(y.shape), [b, out_features])

    @staticmethod
    def _define_input(batch, num_word, word_vec_size):
        return torch.rand(batch, num_word, word_vec_size)

    @staticmethod
    def _define_cnn(in_features=300, out_features=2, filter_windows=(3, 4, 5), num_filter=100, drop_rate=0.5):
        return _CNN(in_features, out_features, filter_windows, num_filter, drop_rate)


class _CNNMultiChannelTest(_CNNTest):
    @staticmethod
    def _define_input(batch, num_word, word_vec_size):
        return torch.rand(batch, num_word, word_vec_size, 2)

    @staticmethod
    def _define_cnn(in_features=300, out_features=2, filter_windows=(3, 4, 5), num_filter=100, drop_rate=0.5):
        return _CNNMultiChannel(in_features, out_features, filter_windows, num_filter, drop_rate)


if __name__ == "__main__":
    unittest.main()
