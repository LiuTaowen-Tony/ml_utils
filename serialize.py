import dataclasses
from typing import Any, Dict, Optional, Type, get_origin, get_args, TypeVar
import json
import os
import numpy as np
import torch

T = TypeVar('T')

# Module-level registry for polymorphic deserialization
_registry: Dict[str, Type] = {}

def register_class(name: str, target_class: Type) -> None:
    _registry[name] = target_class

def get_registered_class(name: str) -> Optional[Type]:
    return _registry.get(name)

def list_registered_classes() -> Dict[str, Type]:
    return _registry.copy()

def as_dict(
    obj, convert_array: bool = True, init_only: bool = True, include_class_info: bool = False
) -> Dict[str, Any]:
    """Convert an object to a dictionary."""
    result = {}
    
    # Add class information if requested and class is registered
    if include_class_info:
        for name, cls in _registry.items():
            if cls is obj.__class__:
                result["__class__"] = name
                break
    
    if dataclasses.is_dataclass(obj):
        data = {}
        for field in dataclasses.fields(obj):
            if init_only and field.init:
                value = getattr(obj, field.name)
                data[field.name] = as_dict(value, convert_array, init_only, include_class_info)
        if include_class_info and result:
            result.update(data)
        else:
            result = data
    elif convert_array and isinstance(obj, np.ndarray):
        result = obj.tolist()
    elif convert_array and isinstance(obj, torch.Tensor):
        result = obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, list):
        result = [as_dict(item, convert_array, init_only, include_class_info) for item in obj]
    elif isinstance(obj, dict):
        result = {
            key: as_dict(value, convert_array, init_only, include_class_info) for key, value in obj.items()
        }
    elif hasattr(obj, "_asdict"):  # Named tuples
        result = {
            key: as_dict(value, convert_array, init_only, include_class_info)
            for key, value in obj._asdict().items()
        }
    elif hasattr(obj, "__dict__"):  # any other classes
        data = {
            key: as_dict(value, convert_array, init_only, include_class_info)
            for key, value in obj.__dict__.items()
        }
        if include_class_info and result:
            result.update(data)
        else:
            result = data
    else:
        result = obj
        
    return result

def from_dict(data: Dict[str, Any], target_class: Optional[Type[T]] = None) -> T:
    """Create an object from a dictionary."""
    # Check if data contains class information
    if isinstance(data, dict) and "__class__" in data:
        class_name = data["__class__"]
        registered_class = get_registered_class(class_name)
        if registered_class is not None:
            target_class = registered_class
            # Remove class info from data for construction
            data = {k: v for k, v in data.items() if k != "__class__"}
    
    assert (
        target_class is not None
    ), "Must provide target_class when calling from_dict, or ensure class info is in data"

    if not dataclasses.is_dataclass(target_class):
        return target_class(data)

    construction_dict = {}
    fields = {f.name: f for f in dataclasses.fields(target_class) if f.init}

    for k, v in data.items():
        if k not in fields:
            continue
        field = fields[k]
        field_type = field.type
        # v is list of dicts and field_type is list of dataclasses
        origin = get_origin(field_type)
        args = get_args(field_type)
        if dataclasses.is_dataclass(field_type) and isinstance(v, dict):
            construction_dict[k] = from_dict(v, field_type)
        elif isinstance(v, list):
            if field_type == np.ndarray:
                construction_dict[k] = np.array(v)
            elif field_type == torch.Tensor:
                construction_dict[k] = torch.tensor(v)
            elif (
                (origin is list) and args and dataclasses.is_dataclass(args[0])
            ):  # field type is parametric list
                construction_dict[k] = [
                    from_dict(item, args[0]) for item in v
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
                k: from_dict(v, args[1]) for k, v in v.items()
            }
        else:
            construction_dict[k] = v
    return target_class(**construction_dict)

def save_json(obj, filepath: str, include_class_info: bool = True) -> None:
    """Save an object to a JSON file."""
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(as_dict(obj, include_class_info=include_class_info), f, indent=4)

def load_json(filepath: str, target_class: Optional[Type[T]] = None) -> T:
    """Load an object from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return from_dict(data, target_class)

def register(name_or_class=None):
    """
    Decorator to register a class for polymorphic deserialization.
    
    Usage:
        @register
        class MyClass:
            pass
            
        @register("CustomName")
        class MyClass:
            pass
    """
    def decorator(cls):
        if isinstance(name_or_class, str):
            registry_name = name_or_class
        else:
            registry_name = cls.__name__
        register_class(registry_name, cls)
        return cls
    
    if name_or_class is None: # Called as @register()
        return decorator
    elif isinstance(name_or_class, str): # Called as @register("name")
        return decorator
    else: # Called as @register (without parentheses)
        return decorator(name_or_class)

