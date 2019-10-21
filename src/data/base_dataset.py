from collections import defaultdict

import constants as const
from utils import fix_random_seed
from data.utils import clean_str


class _BaseDataset(object):
    """
    Load data.
    """
    clean_string = True

    def __init__(self, dataset_file):
        super().__init__()
        self.vocab = defaultdict(int)
        self.data = []
        self._read_datafile(dataset_file)

    def _read_datafile(self, file):
        label_set = set()
        with open(file, 'rb') as f:
            for line in f:
                sentence = line.strip().decode('latin1')
                label = int(sentence.split()[0])
                label_set.add(label)
                sentence = ' '.join(sentence.split()[1:])

                if self.clean_string:
                    orig_sentence = clean_str(sentence)
                else:
                    orig_sentence = sentence.lower()
                words = set(orig_sentence.split())
                for word in words:
                    self.vocab[word] += 1

                datum = {
                    'y': label,
                    'text': orig_sentence,
                    'num_words': len(orig_sentence.split()),
                }
                self.data.append(datum)
        self.num_classes = len(label_set)

    def __len__(self):
        return len(self.data)
