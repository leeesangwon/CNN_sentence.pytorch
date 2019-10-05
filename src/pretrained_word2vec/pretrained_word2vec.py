import numpy as np
import torch


def load_bin_word2vec(fname, vocab):
    """Loads (300,) np.array from Google (Mikolov) word2vec"""
    word_vecs = {}
    with open(fname, 'rb') as f:
        header = f.readline()
        vocab_size, word_vec_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * word_vec_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs, word_vec_size


class PretrainedWord2Vec(object):
    """
    Parameters:
        total_word_list:
        w2v_file: path to `GoogleNews-vectors-negative300.bin`
    """
    def __init__(self, total_word_list, w2v_file):
        word2vec_dict, self.word_vec_size = load_bin_word2vec(w2v_file, total_word_list)

        self.word2index = {}
        self.num_oov = 0
        embeddings = []
        for i, word in enumerate(total_word_list):
            self.word2index[word] = i
            try:
                vec = word2vec_dict[word]
            except KeyError:
                vec = self._unknown_word_embeddings()
                self.num_oov += 1
            embeddings.append(vec)
        embeddings = np.array(embeddings)
        self.embeddings = torch.from_numpy(embeddings)

    def _unknown_word_embeddings(self):
        """
        For OOV words, embedding vectors are initialized from `U[-a, a]`
        where `a` was chosen such that the random vectors have the same variance
        as the pre-trained ones.
        """
        return np.random.uniform(-0.25, 0.25, self.word_vec_size).astype('float32')

    def __len__(self):
        return len(self.word2index)
