from torch.utils.data import Dataset

from .base_dataset import _BaseDataset


class SSTDataset(Dataset):
    def __init__(self, dataset_file):
        super().__init__()
        self._dataset = _BaseDataset(dataset_file, is_sst=True)
        self.vocab = self._dataset.vocab
        self.num_classes = self._dataset.num_classes

    def __getitem__(self, idx):
        label = self._dataset.data[idx]['y']
        sentence = self._dataset.data[idx]['text'].split()
        return sentence, label

    def __len__(self):
        return len(self._dataset)
