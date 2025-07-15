import dataclasses
from typing import Any, Dict, Optional, Type, get_origin, get_args, get_type_hints, TypeVar, Generic
import json
import os
import numpy as np
import torch

T = TypeVar('T', bound='Serial')

class Serial(Generic[T]):

    def as_dict(
        obj, convert_array: bool = True, init_only: bool = True
    ) -> Dict[str, Any]:
        if dataclasses.is_dataclass(obj):
            data = {}
            for field in dataclasses.fields(obj):
                if init_only and field.init:
                    value = getattr(obj, field.name)
                    data[field.name] = Serial.as_dict(value, convert_array)
            return data
        elif convert_array and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif convert_array and isinstance(obj, torch.Tensor): # important is have it before __dict__, because torch.Tensor has __dict__ method
            return obj.cpu().detach().numpy().tolist()
        elif isinstance(obj, list):
            return [Serial.as_dict(item, convert_array) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: Serial.as_dict(value, convert_array) for key, value in obj.items()
            }
        elif hasattr(obj, "_asdict"):  # Named tuples
            return {
                key: Serial.as_dict(value, convert_array)
                for key, value in obj._asdict().items()
            }
        elif hasattr(obj, "__dict__"):  # any other classes
            return {
                key: Serial.as_dict(value, convert_array)
                for key, value in obj.__dict__.items()
            }
        else:
            return obj

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any], target_class: Optional[Type[T]] = None) -> T:
        if cls is Serial:
            assert (
                target_class is not None
            ), "Must provide target_class when calling from_dict on Serial subclass"
            cls = target_class

        if not dataclasses.is_dataclass(cls):
            return cls(data)

        construction_dict = {}
        fields = {f.name: f for f in dataclasses.fields(cls) if f.init}

        for k, v in data.items():
            if k not in fields:
                continue
            field = fields[k]
            field_type = field.type
            # v is list of dicts and field_type is list of dataclasses
            origin = get_origin(field_type)
            args = get_args(field_type)
            if dataclasses.is_dataclass(field_type) and isinstance(v, dict):
                construction_dict[k] = Serial.from_dict(v, field_type)
            elif isinstance(v, list):
                if field_type == np.ndarray:
                    construction_dict[k] = np.array(v)
                elif field_type == torch.Tensor:
                    construction_dict[k] = torch.tensor(v)
                elif (
                    (origin is list) and args and dataclasses.is_dataclass(args[0])
                ):  # field type is parametric list
                    construction_dict[k] = [
                        Serial.from_dict(item, args[0]) for item in v
                    ]
                else:
                    construction_dict[k] = v
            elif (
                isinstance(v, dict)
                and (origin is dict)
                and args
                and dataclasses.is_dataclass(args[1])
            ):  # field type is parametric dict
                construction_dict[k] = {
                    k: Serial.from_dict(v, args[1]) for k, v in v.items()
                }
            else:
                construction_dict[k] = v
        return cls(**construction_dict)

    def save_json(self, filepath: str) -> None:
        """Saves the object to a JSON file."""
        # if filepath is just a name
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        with open(filepath, 'w') as f: # Open in text mode 'w' for JSON
            json.dump(self.as_dict(), f, indent=4)  # Use json.dump and to_dict

    @classmethod
    def load_json(cls: Type[T], filepath: str, target_class: Optional[Type[T]] = None) -> T:
        if cls is Serial:
            cls = target_class
        assert cls is not None, "Must provide target_class when calling from_dict on Serial subclass"
        """Loads the object from a JSON file."""
        with open(filepath, 'r') as f: # Open in text mode 'r' for JSON
            data = json.load(f) # Use json.load
        return cls.from_dict(data) # Use cls.from_dict to reconstruct
