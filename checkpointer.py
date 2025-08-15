"""
FSDP-aware checkpointer for saving components from within FSDP-wrapped modules.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class FSDPComponentCheckpointer:
    """
    Checkpointer designed to save specific components from models (FSDP or non-FSDP).
    
    Automatically detects if the model is FSDP-wrapped and uses the appropriate method:
    - For FSDP models: Uses FSDP.summon_full_params() to safely extract parameters
    - For regular models: Uses standard state_dict() method
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize the checkpointer.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_fsdp_wrapped(self, model: torch.nn.Module) -> bool:
        """
        Check if a model is wrapped with FSDP.
        
        Args:
            model: The model to check
            
        Returns:
            bool: True if the model is FSDP-wrapped
        """
        # Check if the model itself is FSDP
        if isinstance(model, FSDP):
            return True
        
        # Check for nested FSDP modules
        for module in model.modules():
            if isinstance(module, FSDP):
                return True
        
        # Check for Lightning/Fabric wrapped FSDP models
        if hasattr(model, '_forward_module') and isinstance(model._forward_module, FSDP):
            return True
        
        if hasattr(model, 'module') and isinstance(model.module, FSDP):
            return True
            
        return False
    
    def save_component(
        self,
        model: torch.nn.Module,
        component: torch.nn.Module,
        filename: str,
        epoch: int = 0,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a specific component from a model (FSDP or non-FSDP).
        
        Args:
            model: The parent model (may or may not be FSDP-wrapped)
            component: The specific component to save (e.g., model.compressors)
            filename: Name of the checkpoint file
            epoch: Current epoch number
            step: Current step number
            metadata: Additional metadata to save
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            is_fsdp = self._is_fsdp_wrapped(model)
            print(f"Saving component - FSDP detected: {is_fsdp}")
            
            if is_fsdp:
                # Use FSDP.summon_full_params to gather full parameters
                with FSDP.summon_full_params(model, recurse=True):
                    # Extract component state dict while params are available
                    component_state = {}
                    
                    # Get all parameters
                    for name, param in component.named_parameters():
                        component_state[name] = param.detach().clone().cpu()
                    
                    # Get all buffers
                    for name, buffer in component.named_buffers():
                        component_state[name] = buffer.detach().clone().cpu()
            else:
                # Standard model - use regular state_dict
                component_state = component.state_dict()
                # Ensure all tensors are on CPU
                for key, value in component_state.items():
                    if isinstance(value, torch.Tensor):
                        component_state[key] = value.detach().clone().cpu()
            
            # Prepare checkpoint data
            checkpoint = {
                "model": component_state,  # Use "model" key for compatibility
                "epoch": epoch,
                "step": step,
                "metadata": metadata or {},
                "is_fsdp": is_fsdp,  # Record whether this was saved from FSDP
            }
            
            # Save to disk
            checkpoint_path = self.checkpoint_dir / filename
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Component checkpoint saved: {checkpoint_path}")
            print(f"  - Parameters: {len(component_state)}")
            print(f"  - Step: {step}, Epoch: {epoch}")
            print(f"  - FSDP mode: {is_fsdp}")
            
            if component_state:
                sample_param = next(iter(component_state.values()))
                print(f"  - Sample param shape: {sample_param.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error saving component checkpoint: {e}")
            return False
    
    def load_component(
        self,
        model: torch.nn.Module,
        component: torch.nn.Module,
        filename: str,
        strict: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load a component checkpoint into a model (FSDP or non-FSDP).
        
        Args:
            model: The parent model (may or may not be FSDP-wrapped)
            component: The component to load into
            filename: Name of the checkpoint file
            strict: Whether to enforce strict loading
            
        Returns:
            Metadata from the checkpoint if successful, None otherwise
        """
        try:
            checkpoint_path = self.checkpoint_dir / filename
            
            if not checkpoint_path.exists():
                print(f"Checkpoint not found: {checkpoint_path}")
                return None
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            component_state = checkpoint["model"]  # Use "model" key for compatibility
            checkpoint_was_fsdp = checkpoint.get("is_fsdp", True)  # Default to True for backward compatibility
            
            print(f"Loading component checkpoint: {checkpoint_path}")
            print(f"  - Parameters: {len(component_state)}")
            print(f"  - Step: {checkpoint.get('step', 'unknown')}")
            print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  - Checkpoint was from FSDP: {checkpoint_was_fsdp}")
            
            is_fsdp = self._is_fsdp_wrapped(model)
            print(f"  - Current model is FSDP: {is_fsdp}")
            
            if is_fsdp:
                # Load parameters into FSDP module using summon_full_params
                with FSDP.summon_full_params(model, recurse=True):
                    loaded_count = 0
                    missing_params = []
                    
                    for name, param in component.named_parameters():
                        if name in component_state:
                            param.data.copy_(component_state[name].to(param.device))
                            loaded_count += 1
                        else:
                            missing_params.append(name)
                            if strict:
                                raise KeyError(f"Parameter {name} not found in checkpoint")
                    
                    # Load buffers
                    for name, buffer in component.named_buffers():
                        if name in component_state:
                            buffer.data.copy_(component_state[name].to(buffer.device))
                            loaded_count += 1
                        else:
                            missing_params.append(name)
                            if strict:
                                raise KeyError(f"Buffer {name} not found in checkpoint")
            else:
                # Standard model - use load_state_dict
                try:
                    component.load_state_dict(component_state, strict=strict)
                    loaded_count = len(component_state)
                    missing_params = []
                except Exception as e:
                    if strict:
                        raise e
                    # Try to load what we can
                    loaded_count = 0
                    missing_params = []
                    for name, param in component.named_parameters():
                        if name in component_state:
                            param.data.copy_(component_state[name].to(param.device))
                            loaded_count += 1
                        else:
                            missing_params.append(name)
                    
                    for name, buffer in component.named_buffers():
                        if name in component_state:
                            buffer.data.copy_(component_state[name].to(buffer.device))
                            loaded_count += 1
                        else:
                            missing_params.append(name)
            
            print(f"  - Loaded {loaded_count} parameters/buffers")
            if missing_params and not strict:
                print(f"  - Missing (ignored): {missing_params}")
            
            # Return metadata
            return {
                "epoch": checkpoint.get("epoch", 0),
                "step": checkpoint.get("step", 0),
                "metadata": checkpoint.get("metadata", {}),
                "checkpoint_was_fsdp": checkpoint_was_fsdp,
            }
            
        except Exception as e:
            print(f"Error loading component checkpoint: {e}")
            return None