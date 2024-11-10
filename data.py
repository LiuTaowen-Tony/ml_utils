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

    def state_dict(self):
        return {"dataset_state_dict": self.dataset.state_dict(),
                "start": self.start,
                "max_length": self.max_length}

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict["dataset_state_dict"])
        self.start = state_dict["start"]
        self.max_length = state_dict["max_length"]