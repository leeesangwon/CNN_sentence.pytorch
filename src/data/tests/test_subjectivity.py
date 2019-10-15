import os
import unittest

import numpy as np

from ..subjectivity import Subjectivity, _Subjectivity

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '../../../resource/Subj')
np.random.seed(1905)


class SubjectivityTest(unittest.TestCase):
    def test_train_set(self):
        subjectivity = Subjectivity(DATASET_FOLDER,  batch_size=1, test_cv=0, type='train')
        self._test_dataset(subjectivity)

    def test_val_set(self):
        subjectivity = Subjectivity(DATASET_FOLDER,  batch_size=1, test_cv=0, type='val')
        self._test_dataset(subjectivity)

    def test_test_set(self):
        subjectivity = Subjectivity(DATASET_FOLDER,  batch_size=1, test_cv=0, type='test')
        self._test_dataset(subjectivity)

    def _test_dataset(self, dataset):
        sentence_to_check = np.random.randint(0, len(dataset), 10)
        for i in sentence_to_check:
            sentence, label = dataset[i]
            print(label, sentence)
            self._check_sentence_type(sentence)

    def _check_sentence_type(self, sentence):
        self.assertEqual(type(sentence), list)
        for word in sentence:
            self.assertEqual(type(word), str)


class _SubjectivityTest(unittest.TestCase):
    def test_len_data(self):
        _subjectivity = _Subjectivity(DATASET_FOLDER)
        self.assertEqual(len(_subjectivity), 10000)

    def test_vocab(self):
        _subjectivity = _Subjectivity(DATASET_FOLDER)

        print()
        print('len(vocab): ', len(_subjectivity.vocab))
        print('min(vocab): ', min(_subjectivity.vocab.values()))
        print('max(vocab): ', max(_subjectivity.vocab.values()))

        for i, (vocab, freq) in enumerate(_subjectivity.vocab.items()):
            if i == 10:
                break
            print(vocab, freq)
