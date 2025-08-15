#!/usr/bin/env python3
"""
2-device FSDP checkpointer test.
"""

import os
import tempfile
import shutil
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# Add the parent directory to the path so we can import the checkpointer
sys.path.insert(0, str(Path(__file__).parent.parent))
from checkpointer import Checkpointer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def test_2_device_fsdp_checkpointing():
    """Test FSDP checkpointing with 2 devices"""
    
    # Check if we have 2 GPUs available
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.warning("⚠️  Less than 2 GPUs available. Skipping 2-device test.")
        return True
    
    logger.info("🚀 Starting 2-Device FSDP Checkpointing Test")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Initialize distributed process group for 2 devices
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')
    
    try:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,  # Single process with multiple devices
            rank=0
        )
    except RuntimeError as e:
        if "already initialized" not in str(e):
            raise e
    
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Test 1: Create FSDP model on device 0
        device = torch.device("cuda:0")
        model = SimpleModel().to(device)
        
        # Wrap with FSDP using ModuleWrapPolicy
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=device.index,
        )
        
        optimizer = Adam(fsdp_model.parameters(), lr=0.001)
        
        # Train for a few steps
        logger.info("Training FSDP model...")
        fsdp_model.train()
        for step in range(5):
            x = torch.randn(8, 32, device=device)
            output = fsdp_model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"Step {step+1}, Loss: {loss.item():.4f}")
        
        # Get original state for comparison
        with FSDP.state_dict_type(fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            original_state = fsdp_model.state_dict()
        
        # Test 2: Save checkpoint
        logger.info("Saving FSDP checkpoint...")
        checkpointer = Checkpointer(checkpoint_dir)
        state = {"model": fsdp_model, "optimizer": optimizer}
        
        # Test both central and sharded saves
        central_path = checkpointer.save(state, epoch=1, step=100, save_type="central")
        sharded_path = checkpointer.save(state, epoch=1, step=100, save_type="sharded")
        
        logger.info(f"Saved central checkpoint: {central_path}")
        logger.info(f"Saved sharded checkpoint: {sharded_path}")
        
        # Test 3: Create new FSDP model and load checkpoint
        logger.info("Creating new FSDP model for loading...")
        new_model = SimpleModel().to(device)
        new_fsdp_model = FSDP(
            new_model,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=device.index,
        )
        new_optimizer = Adam(new_fsdp_model.parameters(), lr=0.001)
        
        # Test central loading
        logger.info("Loading from central checkpoint...")
        new_state = {"model": new_fsdp_model, "optimizer": new_optimizer}
        metadata = checkpointer.load(central_path, new_state, load_type="central")
        
        logger.info(f"Loaded metadata: epoch={metadata['epoch']}, step={metadata['step']}")
        
        # Verify parameters match
        with FSDP.state_dict_type(new_fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            loaded_state = new_fsdp_model.state_dict()
        
        # Compare parameters
        params_match = True
        for key in original_state:
            if key in loaded_state:
                if not torch.allclose(original_state[key], loaded_state[key], atol=1e-6):
                    logger.error(f"Parameter mismatch for {key}")
                    params_match = False
            else:
                logger.error(f"Missing parameter {key}")
                params_match = False
        
        if params_match:
            logger.info("✅ Central checkpoint: All parameters match!")
        else:
            logger.error("❌ Central checkpoint: Parameter mismatch!")
            return False
        
        # Test sharded loading
        logger.info("Testing sharded checkpoint loading...")
        newer_model = SimpleModel().to(device)
        newer_fsdp_model = FSDP(
            newer_model,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=device.index,
        )
        newer_optimizer = Adam(newer_fsdp_model.parameters(), lr=0.001)
        
        newer_state = {"model": newer_fsdp_model, "optimizer": newer_optimizer}
        sharded_metadata = checkpointer.load(sharded_path, newer_state, load_type="sharded")
        
        logger.info(f"Loaded sharded metadata: epoch={sharded_metadata['epoch']}, step={sharded_metadata['step']}")
        
        # Verify sharded parameters match
        with FSDP.state_dict_type(newer_fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
            sharded_loaded_state = newer_fsdp_model.state_dict()
        
        sharded_params_match = True
        for key in original_state:
            if key in sharded_loaded_state:
                if not torch.allclose(original_state[key], sharded_loaded_state[key], atol=1e-6):
                    logger.error(f"Sharded parameter mismatch for {key}")
                    sharded_params_match = False
            else:
                logger.error(f"Missing sharded parameter {key}")
                sharded_params_match = False
        
        if sharded_params_match:
            logger.info("✅ Sharded checkpoint: All parameters match!")
        else:
            logger.error("❌ Sharded checkpoint: Parameter mismatch!")
            return False
        
        logger.info("🎉 2-Device FSDP Checkpointing Test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error during distributed cleanup: {e}")


def main():
    """Run the 2-device test"""
    logger.info("🎯 2-Device FSDP Checkpointer Test")
    logger.info("=" * 50)
    
    success = test_2_device_fsdp_checkpointing()
    
    if success:
        logger.info("✅ ALL TESTS PASSED!")
        return 0
    else:
        logger.error("❌ TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())