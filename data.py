import torch
from torch.utils.data import Dataset, IterableDataset

class IterableSubset(IterableDataset):
    def __init__(self, dataset: Dataset, max_length: int, start: int = 0):
        self.dataset = dataset
        self.start = start
        self.max_length = max_length

    def __iter__(self):
        it = iter(self.dataset)
        for _ in range(self.start):
            next(it)
        for _ in range(self.max_length):
            yield next(it)

    def __len__(self):
        return self.max_length

