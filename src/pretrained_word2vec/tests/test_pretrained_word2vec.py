import os
import unittest

from ..pretrained_word2vec import PretrainedWord2Vec

W2V_FILE = os.path.join(os.path.dirname(__file__), '../GoogleNews-vectors-negative300.bin')


class PretrainedWord2VecTest(unittest.TestCase):
    def test_oov(self):
        word_list = ['deep', 'learning', 'sentence', 'classification']
        oov_list = ['jjfdks', 'fadfdsa']
        pretrained_word2vec = PretrainedWord2Vec(word_list + oov_list, w2v_file=W2V_FILE)

        self.assertEqual(pretrained_word2vec.embeddings.size(0), len(word_list+oov_list))
        self.assertEqual(pretrained_word2vec.embeddings.size(1), pretrained_word2vec.word_vec_size)
        self.assertEqual(pretrained_word2vec.num_oov, len(oov_list))


if __name__ == "__main__":
    unittest.main()
