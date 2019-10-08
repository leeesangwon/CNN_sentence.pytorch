import os
import unittest

import numpy as np

from ..movie_reviews import MovieReview, _MovieReview

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), '../../../data/MR')
np.random.seed(1905)


class MovieReviewTest(unittest.TestCase):
    def test_train_set(self):
        movie_review = MovieReview(DATASET_FOLDER, val_cv=0, is_val=False)
        sentence_to_check = np.random.randint(0, len(movie_review), 10)
        for i in sentence_to_check:
            sentence, label = movie_review[i]
            print(label, sentence)
            self._check_sentence_type(sentence)

    def test_val_set(self):
        movie_review = MovieReview(DATASET_FOLDER, val_cv=0, is_val=True)
        sentence_to_check = np.random.randint(0, len(movie_review), 10)
        for i in sentence_to_check:
            sentence, label = movie_review[i]
            print(label, sentence)
            self._check_sentence_type(sentence)

    def _check_sentence_type(self, sentence):
        self.assertEqual(type(sentence), list)
        for word in sentence:
            self.assertEqual(type(word), str)


class _MovieReviewTest(unittest.TestCase):
    def test_len_data(self):
        _movie_review = _MovieReview(DATASET_FOLDER)
        self.assertEqual(len(_movie_review), 10662)
        self.assertEqual(len(_movie_review.data), _movie_review.cv)
        print('length of each fold:')
        for data in _movie_review.data:
            print(len(data), end='\t')

    def test_vocab(self):
        _movie_review = _MovieReview(DATASET_FOLDER)

        print()
        print('len(vocab): ', len(_movie_review.vocab))
        print('min(vocab): ', min(_movie_review.vocab.values()))
        print('max(vocab): ', max(_movie_review.vocab.values()))

        for i, (vocab, freq) in enumerate(_movie_review.vocab.items()):
            if i == 10:
                break
            print(vocab, freq)
