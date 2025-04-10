import json
import inspect
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import torchvision

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

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



class GPUDataset(Dataset):
    def __init__(self, data, label, device="cuda"):
        self.data = data.to(device)
        self.label = label.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    @staticmethod
    def get_dataset_by_name(cls, name):
        if name == "cifar10":
            train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
            test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
            ret = [torch.tensor(i) for i in (train.data, train.targets, test.data, test.targets)]
            std, mean = torch.std_mean(ret[0].float(), dim=(0, 1, 2), unbiased=True, keepdim=True)
            for i in [0, 2]:
                ret[i] = ((ret[i] - mean) / std).to(torch.float32).permute(0, 3, 1, 2)
            return GPUDataset(ret[0], ret[1]), GPUDataset(ret[2], ret[3])
        elif "mnist" in name:
            if "fashion" in name:
                dataset = torchvision.datasets.FashionMNIST
            elif "colored" in name:
                dataset = torchvision.datasets.MNIST
            else:
                dataset = torchvision.datasets.MNIST
            train = dataset(root="./data", train=True, download=True)
            test = dataset(root="./data", train=False, download=True)
            ret = [torch.tensor(i) for i in (train.data, train.targets, test.data, test.targets)]
            for i in [0, 2]:
                ret[i] = ret[i].float().to("cuda").view(-1, 1, 28, 28)
            return GPUDataset(ret[0], ret[1]), GPUDataset(ret[2], ret[3])
        

class LiftedTransform:
    """
    Enhanced lift function with input mapping support.
        
        Args:
            transform: Function to lift (multi-input multi-output)
            output_keys: Output keys in result dictionary
            default_params: Default parameters
            input_mapping: Dict mapping parameter names to item keys
        
        # Example usage with input mapping:
        def combine_texts(first_text, second_text, separator=" "):
            return f"{first_text}{separator}{second_text}"

        lifted_combiner = lift(
            combine_texts,
            "combined_text",
            default_params={"separator": " | "},
            input_mapping={
                "first_text": "title",
                "second_text": "description"
            }
        )

        result = lifted_combiner({
            "title": "Hello",
            "description": "World"
        })
        # Result: {"title": "Hello", "description": "World", "combined_text": "Hello | World"}
    """
    def __init__(self, output_key = None,  default_params = None, input_mapping = None):
        self.default_params = default_params or {"self": self}
        self.output_key = output_key
        self.input_mapping = input_mapping or {}

    def transform(self, **kwargs):
        raise NotImplementedError

    def __call__(self, item):
        transform = self.transform
        params = inspect.signature(transform).parameters

        inputs = {}
        for param_name, param in params.items():
            # Check if there's a mapping for this parameter
            item_key = self.input_mapping.get(param_name, param_name)
            
            if item_key in item:
                inputs[param_name] = item[item_key]
            elif param_name in self.default_params:
                inputs[param_name] = self.default_params[param_name]
            elif param.default != inspect.Parameter.empty:
                raise KeyError(f"Required parameter '{param_name}' not found in item or defaults")
        
        output = transform(**inputs)
        if self.output_key == None:
            item.update(output)
            return item
        else:
            item[self.output_key] = output
            return item




# def batch_lifted(transform, output_key, default_params=None, input_mapping=None):
#     """
#     Batch version of the lift function.
#     Work on dict of lists.
    
#     Args:
#         transform: Function to lift
#         output_key: Output key in result dictionary
#         default_params: Default parameters
#         input_mapping: Dict mapping parameter names to item keys
#     """
#     @functools.wraps(transform)
#     def batch_lifted_transform(batch):
#         # 1. construct the lifted transform
#         lifted_transform = lift(transform, output_key, default_params, input_mapping)
#         # 2. convert dict of lists to list of dicts
#         list_of_items = []
#         keys = batch.keys()
        
#         # 2.2. get the length of the first list
#         length = len(batch[keys[0]])
        
#         # 2.3. iterate over the length
#         for i in range(length):
#             # 2.3.1. construct the item
#             item = {key: batch[key][i] for key in keys}
#             list_of_items.append(item)
        
#         # 3. apply the lifted transform to each item
#         list_of_results = [lifted_transform(item) for item in list_of_items]

#         # 4. convert list of dicts to dict of lists
#         results = {key: [] for key in list_of_results[0].keys()}
#         for result in list_of_results:
#             for key in results.keys():
#                 results[key].append(result[key])

#         return results
#     return batch_lifted_transformimport torch

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
