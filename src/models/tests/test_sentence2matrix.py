import unittest

from ..sentence2matrix import Sentence2Mat
from src.pretrained_word2vec import PretrainedWord2Vec
from src.pretrained_word2vec.tests.test_pretrained_word2vec import W2V_FILE


class Sentence2MatTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_word2vec = self._generate_dummy_pretrained()
        self.WORD_VEC_SIZE = self.pretrained_word2vec.word_vec_size

    def test_random(self):
        word_list = self._generate_word_list()
        word2vec_ = self._generate_word2vec(False, False)
        embeddings = word2vec_(word_list)
        self.assertEqual(list(embeddings.shape), [len(word_list), self.WORD_VEC_SIZE])

    def test_non_static(self):
        word_list = self._generate_word_list()
        word2vec_ = self._generate_word2vec(True, False)
        embeddings = word2vec_(word_list)
        self.assertEqual(list(embeddings.shape), [len(word_list), self.WORD_VEC_SIZE])

    def test_static(self):
        word_list = self._generate_word_list()
        word2vec_ = self._generate_word2vec(True, True)
        embeddings = word2vec_(word_list)
        self.assertEqual(list(embeddings.shape), [len(word_list), self.WORD_VEC_SIZE])

    def _generate_word2vec(self, use_pretrained, freeze):
        return Sentence2Mat(self.pretrained_word2vec, use_pretrained, freeze)

    def _generate_dummy_pretrained(self):
        word_list = self._generate_word_list()
        return PretrainedWord2Vec(word_list, W2V_FILE)

    @staticmethod
    def _generate_word_list():
        return ['hi', 'bye', 'good']
