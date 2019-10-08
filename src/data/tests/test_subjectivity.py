import os
import unittest

import numpy as np

from ..subjectivity import Subjectivity, _Subjectivity

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '../../../data/Subj')
np.random.seed(1905)


class SubjectivityTest(unittest.TestCase):
    def test_train_set(self):
        subjectivity = Subjectivity(DATASET_FOLDER, val_cv=0, is_val=False)
        sentence_to_check = np.random.randint(0, len(subjectivity), 10)
        for i in sentence_to_check:
            sentence, label = subjectivity[i]
            print(label, sentence)
            self._check_sentence_type(sentence)

    def test_val_set(self):
        subjectivity = Subjectivity(DATASET_FOLDER, val_cv=0, is_val=True)
        sentence_to_check = np.random.randint(0, len(subjectivity), 10)
        for i in sentence_to_check:
            sentence, label = subjectivity[i]
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
        self.assertEqual(len(_subjectivity.data), _subjectivity.cv)
        print('length of each fold:')
        for data in _subjectivity.data:
            print(len(data), end='\t')

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
