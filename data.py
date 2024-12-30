import inspect
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
#     return batch_lifted_transform