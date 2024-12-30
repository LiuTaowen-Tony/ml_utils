import json
import os
import pathlib
import torch

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def relative_path(file_path: str, underscore_file) -> str:
    return (pathlib.Path(underscore_file).parent / file_path).as_posix()

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def save_with_config(model: torch.nn.Module, path: str, config: dict = None):
    """Save model with configuration."""
    os.makedirs(path, exist_ok=True)
    if config is None:
        if hasattr(model, "config"):
            config = model.config
        else:
            print("No configuration found.")
    with open(os.path.join(path, "config.json"), "w") as f:
        if isinstance(config, dict):
            json.dump(config, f)
        else:
            json.dump(config.__dict__, f)
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))

def load_with_config(config_class, model_class, path: str):
    """Load model with configuration."""
    with open(os.join(path, "config.json"), "r") as f:
        config = config_class(**json.load(f))
    model = model_class(**config, config=config)
    model.load_state_dict(torch.load(os.path.join(path, "model.pth")))
    return model