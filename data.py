import torch
from torch.utils.data import Dataset, IterableDataset

class IterableSubset(IterableDataset):
    def __init__(self, dataset: Dataset, max_length: int, start: int = 0):
        self.dataset = dataset

    def __iter__(self):
        for i in range(len(self.dataset)):
            if i >= self.start + self.max_length:
                break
            yield self.dataset[i]

    def __len__(self):
        return min(len(self.dataset) - self.start, self.max_length)

