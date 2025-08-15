#!/usr/bin/env python3
"""
Standalone checkpointer tests without pytest.
Tests all FSDP checkpointing scenarios.

These tests are configured to use 2 devices for FSDP testing where applicable.
For true multi-device FSDP testing with proper sharding, use the provided
run_fsdp_tests.py script which uses torchrun.

Usage: python test_checkpointer_standalone.py
"""

import os
import tempfile
import shutil
import logging
import socket
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy

# Add the parent directory to the path so we can import the checkpointer
sys.path.insert(0, str(Path(__file__).parent.parent))
from checkpointer import Checkpointer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test Models
class Backbone(nn.Module):
    def __init__(self, input_size=32, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Linear(64, 10)
        self.adapter = nn.Linear(64, 32)
    
    def forward(self, x):
        features = self.backbone(x)
        head_out = self.head(features)
        adapter_out = self.adapter(features)
        return head_out + adapter_out.mean(dim=-1, keepdim=True).expand_as(head_out)


class TestRunner:
    """Test runner for checkpointer scenarios"""
    
    def __init__(self):
        self.temp_dir = None
        self.checkpoint_dir = None
        self.setup()
    
    def _find_free_port(self):
        """Find a free port for distributed training"""
        # Allow override via environment variable for debugging
        if 'PYTORCH_DIST_PORT' in os.environ:
            return int(os.environ['PYTORCH_DIST_PORT'])
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _verify_model_params_equal(self, original_state_dict, loaded_model, test_name):
        """Verify that loaded model parameters match original parameters"""
        loaded_state_dict = loaded_model.state_dict()
        
        # Compare all parameters
        for key in original_state_dict:
            if key in loaded_state_dict:
                try:
                    torch.testing.assert_close(
                        original_state_dict[key], 
                        loaded_state_dict[key],
                        msg=f"{test_name}: Parameter {key} mismatch"
                    )
                except Exception as e:
                    logger.error(f"{test_name}: Parameter verification failed for {key}: {e}")
                    return False
            else:
                logger.error(f"{test_name}: Parameter {key} missing in loaded model")
                return False
        
        logger.info(f"{test_name}: ✅ All parameters match!")
        return True
    
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize distributed if not already done
        # Note: When using torchrun, distributed is already initialized
        if not torch.distributed.is_initialized():
            free_port = self._find_free_port()
            # For single-process testing, use world_size=1
            # For true multi-device testing, use torchrun which handles initialization
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=f"tcp://127.0.0.1:{free_port}",
                world_size=1,
                rank=0
            )
        
        # Check if we're running with torchrun (multi-device)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        if world_size > 1:
            logger.info(f"Running with torchrun: world_size={world_size}, rank={rank}")
        else:
            logger.info("Running in single-process mode (FSDP will use NO_SHARD)")
        
        logger.info(f"Test setup complete. Checkpoint dir: {self.checkpoint_dir}")
    
    def teardown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up distributed if initialized
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                logger.warning(f"Error during distributed cleanup: {e}")
                # Force cleanup by setting the global state
                torch.distributed.distributed_c10d._world = None
                torch.distributed.distributed_c10d._default_pg = None
    
    def create_normal_model(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Create a normal (non-FSDP) model"""
        model = SimpleModel().to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        return model, optimizer
    
    def create_fsdp_model(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Create an FSDP-wrapped model"""
        model = SimpleModel().to(device)
        
        fsdp_model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({Backbone}))
        optimizer = Adam(fsdp_model.parameters(), lr=0.001)
        return fsdp_model, optimizer
    
    def create_fabric_fsdp_model(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Create an FSDP-wrapped model using Fabric strategy"""
        # Determine number of devices based on distributed setup
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        num_devices = world_size if world_size > 1 else 2  # Use 2 for single-process, actual world_size for torchrun
        
        # Create Fabric with FSDP strategy
        fabric = Fabric(
            devices=num_devices, 
            strategy=FSDPStrategy(auto_wrap_policy=ModuleWrapPolicy({Backbone}))
        )
        fabric.launch()
        
        # Create model and optimizer
        model = SimpleModel().to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        # Setup with Fabric (this applies FSDP)
        model, optimizer = fabric.setup(model, optimizer)
        
        return model, optimizer, fabric
    
    def train_model(self, model, optimizer, device="cuda" if torch.cuda.is_available() else "cpu", fabric=None):
        """Train model for a few steps"""
        model.train()
        for _ in range(3):
            x = torch.randn(16, 32, device=device)
            loss = model(x).sum()
            
            # Use fabric.backward if fabric is provided, otherwise regular backward
            if fabric is not None:
                fabric.backward(loss)
            else:
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()


def test_1_baseline_model_save_load(runner):
    """Test 1: Baseline normal model save/load"""
    logger.info("🚀 Starting Test 1: Baseline model save/load (central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and train normal model
    model, optimizer = runner.create_normal_model(device)
    runner.train_model(model, optimizer, device)
    
    # Store original state for comparison
    original_state = model.state_dict()
    
    # Save checkpoint using central type (single file)
    checkpointer = Checkpointer(runner.checkpoint_dir)
    state = {"model": model, "optimizer": optimizer}
    checkpoint_path = checkpointer.save(state, epoch=1, step=100, save_type="central")
    
    # Load into new model
    new_model, new_optimizer = runner.create_normal_model(device)
    new_state = {"model": new_model, "optimizer": new_optimizer}
    metadata = checkpointer.load(checkpoint_path, new_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 1
    assert metadata["step"] == 100
    
    # Verify parameters match
    success = runner._verify_model_params_equal(original_state, new_model, "Test 1")
    assert success, "Parameter verification failed"
    
    logger.info("✅ Test 1: Baseline model save/load (central) passed!")
    return True


def test_2_fsdp_save_normal_load_no_fabric(runner):
    """Test 2: FSDP model save, normal model load (no fabric)"""
    logger.info("🚀 Starting Test 2: FSDP save -> normal load (no fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and train FSDP model
    fsdp_model, fsdp_optimizer = runner.create_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device)
    
    # Store original state for comparison (get full state dict from FSDP)
    with FSDP.state_dict_type(fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        original_state = fsdp_model.state_dict()
    
    # Save FSDP checkpoint using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_state = {"model": fsdp_model, "optimizer": fsdp_optimizer}
    checkpoint_path = checkpointer.save(fsdp_state, epoch=2, step=200, save_type="central")
    
    # Load into normal model
    normal_model, normal_optimizer = runner.create_normal_model(device)
    normal_state = {"model": normal_model, "optimizer": normal_optimizer}
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 2
    assert metadata["step"] == 200
    
    # Verify parameters match
    success = runner._verify_model_params_equal(original_state, normal_model, "Test 2")
    assert success, "FSDP -> Normal parameter verification failed"
    
    logger.info("✅ Test 2: FSDP save -> normal load (no fabric, central) passed!")
    return True


def test_3_non_fsdp_partial_save_load_no_fabric(runner):
    """Test 3: Non-FSDP partial model save/load (no fabric)"""
    logger.info("🚀 Starting Test 3: Non-FSDP partial save -> normal load (no fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create full model
    model, _ = runner.create_normal_model(device)
    
    # Train the full model
    optimizer = Adam(model.parameters(), lr=0.001)
    runner.train_model(model, optimizer, device)
    
    # Store original state for comparison
    original_head_state = model.head.state_dict()
    original_adapter_state = model.adapter.state_dict()
    
    # Save only non-FSDP parts (head and adapter) using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    partial_state = {
        "head": model.head,
        "adapter": model.adapter,
    }
    checkpoint_path = checkpointer.save(partial_state, epoch=3, step=300, save_type="central")
    
    # Load into normal model parts
    normal_model, _ = runner.create_normal_model()
    normal_model = normal_model.to(device)
    
    normal_state = {
        "head": normal_model.head,
        "adapter": normal_model.adapter,
    }
    
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 3
    assert metadata["step"] == 300
    
    # Verify parameters match for head and adapter
    loaded_head_state = normal_model.head.state_dict()
    loaded_adapter_state = normal_model.adapter.state_dict()
    
    head_match = all(torch.allclose(original_head_state[k], loaded_head_state[k]) for k in original_head_state)
    adapter_match = all(torch.allclose(original_adapter_state[k], loaded_adapter_state[k]) for k in original_adapter_state)
    
    assert head_match and adapter_match, "Partial parameter verification failed"
    
    logger.info("✅ Test 3: Non-FSDP partial save -> normal load (no fabric, central) passed!")
    return True


def test_4_fsdp_partial_save_normal_load_no_fabric(runner):
    """Test 4: FSDP partial model (FSDP part), normal load (no fabric)"""
    logger.info("🚀 Starting Test 4: FSDP partial (FSDP) save -> normal load (no fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create FSDP model
    fsdp_model, fsdp_optimizer = runner.create_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device)
    
    # Store original backbone state for comparison
    with FSDP.state_dict_type(fsdp_model.backbone, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        original_backbone_state = fsdp_model.backbone.state_dict()
    
    # Extract FSDP part (backbone is FSDP-wrapped)
    fsdp_backbone = fsdp_model.backbone
    
    # Save FSDP part using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_partial_state = {
        "backbone": fsdp_backbone,
    }
    checkpoint_path = checkpointer.save(fsdp_partial_state, epoch=4, step=400, save_type="central")
    
    # Load into normal model backbone
    normal_model, _ = runner.create_normal_model()
    normal_model = normal_model.to(device)
    
    normal_state = {
        "backbone": normal_model.backbone,
    }
    
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 4
    assert metadata["step"] == 400
    
    # Verify parameters match for backbone
    loaded_backbone_state = normal_model.backbone.state_dict()
    backbone_match = all(torch.allclose(original_backbone_state[k], loaded_backbone_state[k]) for k in original_backbone_state)
    
    assert backbone_match, "FSDP partial parameter verification failed"
    
    logger.info("✅ Test 4: FSDP partial (FSDP) save -> normal load (no fabric, central) passed!")
    return True


def test_5_fsdp_save_normal_load_with_fabric(runner):
    """Test 5: FSDP model save, normal model load (with fabric)"""
    logger.info("🚀 Starting Test 5: FSDP save -> normal load (with fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and train FSDP model using Fabric strategy (Fabric handles distributed setup)
    fsdp_model, fsdp_optimizer, fsdp_fabric = runner.create_fabric_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device, fsdp_fabric)
    
    # Store original state for comparison
    original_state = fsdp_model.state_dict()
    
    # Save FSDP checkpoint with fabric using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_state = {"model": fsdp_model, "optimizer": fsdp_optimizer}
    checkpoint_path = checkpointer.save(fsdp_state, epoch=5, step=500, save_type="central")
    
    # Create Fabric instance for loading (Fabric handles setup automatically)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    num_devices = world_size if world_size > 1 else 2
    load_fabric = Fabric(devices=num_devices, strategy="auto")
    load_fabric.launch()
    
    # Load into normal model with fabric
    normal_model, normal_optimizer = runner.create_normal_model(device)
    normal_model, normal_optimizer = load_fabric.setup(normal_model, normal_optimizer)
    
    normal_state = {"model": normal_model, "optimizer": normal_optimizer}
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 5
    assert metadata["step"] == 500
    
    # Verify parameters match
    success = runner._verify_model_params_equal(original_state, normal_model, "Test 5")
    assert success, "Parameter verification failed"
    
    logger.info("✅ Test 5: FSDP save -> normal load (with fabric, central) passed!")
    return True


def test_6_non_fsdp_partial_save_load_with_fabric(runner):
    """Test 6: Non-FSDP partial model save, normal model load (with fabric)"""
    logger.info("🚀 Starting Test 6: Non-FSDP partial save -> normal load (with fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create Fabric instance (handles distributed setup automatically)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    num_devices = world_size if world_size > 1 else 2
    fabric = Fabric(devices=num_devices, strategy="auto")
    fabric.launch()
    
    # Create full model
    model, _ = runner.create_normal_model(device)
    
    # Train the full model
    optimizer = Adam(model.parameters(), lr=0.001)
    runner.train_model(model, optimizer, device)
    
    # Store original state for comparison
    original_head_state = model.head.state_dict()
    original_adapter_state = model.adapter.state_dict()
    
    # Setup non-FSDP parts with fabric
    head_component, _ = fabric.setup(model.head, Adam(model.head.parameters()))
    adapter_component, _ = fabric.setup(model.adapter, Adam(model.adapter.parameters()))
    
    # Save with fabric using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    partial_state = {
        "head": head_component,
        "adapter": adapter_component,
    }
    checkpoint_path = checkpointer.save(partial_state, epoch=6, step=600, save_type="central")
    
    # Load into normal model with fabric
    normal_model, _ = runner.create_normal_model()
    normal_model.head, _ = fabric.setup(normal_model.head, Adam(normal_model.head.parameters()))
    normal_model.adapter, _ = fabric.setup(normal_model.adapter, Adam(normal_model.adapter.parameters()))
    
    normal_state = {
        "head": normal_model.head,
        "adapter": normal_model.adapter,
    }
    
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 6
    assert metadata["step"] == 600
    
    # Verify parameters match
    loaded_head_state = normal_model.head.state_dict()
    loaded_adapter_state = normal_model.adapter.state_dict()
    
    head_match = all(torch.allclose(original_head_state[k], loaded_head_state[k]) for k in original_head_state)
    adapter_match = all(torch.allclose(original_adapter_state[k], loaded_adapter_state[k]) for k in original_adapter_state)
    
    assert head_match and adapter_match, "Partial parameter verification failed"
    
    logger.info("✅ Test 6: Non-FSDP partial save -> normal load (with fabric, central) passed!")
    return True


def test_7_fsdp_partial_fsdp_save_normal_load_with_fabric(runner):
    """Test 7: FSDP partial model (FSDP part), load (with fabric)"""
    logger.info("🚀 Starting Test 7: FSDP partial (FSDP) save -> normal load (with fabric, central)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create FSDP model using Fabric strategy (Fabric handles distributed setup)
    fsdp_model, fsdp_optimizer, fsdp_fabric = runner.create_fabric_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device, fsdp_fabric)
    
    # Store original backbone state for comparison
    original_backbone_state = fsdp_model.backbone.state_dict()
    
    # Extract FSDP part (backbone)
    fsdp_backbone = fsdp_model.backbone
    
    # Save with fabric using central type
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_partial_state = {
        "backbone": fsdp_backbone,
    }
    checkpoint_path = checkpointer.save(fsdp_partial_state, epoch=7, step=700, save_type="central")
    
    # Create Fabric instance for loading (Fabric handles setup automatically)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    num_devices = world_size if world_size > 1 else 2
    load_fabric = Fabric(devices=num_devices, strategy="auto")
    load_fabric.launch()
    
    # Load into normal model with fabric
    normal_model, _ = runner.create_normal_model()
    normal_model.backbone, _ = load_fabric.setup(normal_model.backbone, Adam(normal_model.backbone.parameters()))
    
    normal_state = {
        "backbone": normal_model.backbone,
    }
    
    metadata = checkpointer.load(checkpoint_path, normal_state, load_type="central")
    
    # Verify metadata
    assert metadata["epoch"] == 7
    assert metadata["step"] == 700
    
    # Verify parameters match
    loaded_backbone_state = normal_model.backbone.state_dict()
    backbone_match = all(torch.allclose(original_backbone_state[k], loaded_backbone_state[k]) for k in original_backbone_state)
    
    assert backbone_match, "FSDP Fabric partial parameter verification failed"
    
    logger.info("✅ Test 7: FSDP partial (FSDP) save -> normal load (with fabric, central) passed!")
    return True


def test_8_fsdp_sharded_save_load(runner):
    """Test 8: FSDP sharded checkpoint save/load"""
    logger.info("🚀 Starting Test 8: FSDP sharded save/load")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and train FSDP model
    fsdp_model, fsdp_optimizer = runner.create_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device)
    
    # Store original state for comparison
    with FSDP.state_dict_type(fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        original_state = fsdp_model.state_dict()
    
    # Save FSDP checkpoint using sharded type (multiple files, one per rank)
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_state = {"model": fsdp_model, "optimizer": fsdp_optimizer}
    checkpoint_path = checkpointer.save(fsdp_state, epoch=8, step=800, save_type="sharded")
    
    # Load into new FSDP model using sharded type
    new_fsdp_model, new_fsdp_optimizer = runner.create_fsdp_model(device)
    new_fsdp_state = {"model": new_fsdp_model, "optimizer": new_fsdp_optimizer}
    metadata = checkpointer.load(checkpoint_path, new_fsdp_state, load_type="sharded")
    
    # Verify metadata
    assert metadata["epoch"] == 8
    assert metadata["step"] == 800
    
    # Verify parameters match
    with FSDP.state_dict_type(new_fsdp_model, torch.distributed.fsdp.StateDictType.FULL_STATE_DICT):
        loaded_state = new_fsdp_model.state_dict()
    
    success = all(torch.allclose(original_state[k], loaded_state[k]) for k in original_state)
    assert success, "FSDP sharded parameter verification failed"
    
    logger.info("✅ Test 8: FSDP sharded save/load passed!")
    return True


def test_9_fsdp_fabric_sharded_save_load(runner):
    """Test 9: FSDP sharded checkpoint with Fabric"""
    logger.info("🚀 Starting Test 9: FSDP sharded save/load with Fabric")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create and train FSDP model using Fabric (handles distributed setup automatically)
    fsdp_model, fsdp_optimizer, fsdp_fabric = runner.create_fabric_fsdp_model(device)
    runner.train_model(fsdp_model, fsdp_optimizer, device, fsdp_fabric)
    
    # Store original state for comparison
    original_state = fsdp_model.state_dict()
    
    # Save FSDP checkpoint using sharded type with Fabric
    checkpointer = Checkpointer(runner.checkpoint_dir)
    fsdp_state = {"model": fsdp_model, "optimizer": fsdp_optimizer}
    checkpoint_path = checkpointer.save(fsdp_state, epoch=9, step=900, save_type="sharded")
    
    # Create new Fabric instance for loading (handles setup automatically)
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    num_devices = world_size if world_size > 1 else 2
    load_fabric = Fabric(
        devices=num_devices, 
        strategy=FSDPStrategy(auto_wrap_policy=ModuleWrapPolicy({Backbone}))
    )
    load_fabric.launch()
    
    # Load into new FSDP model with Fabric
    new_model = SimpleModel().to(device)
    new_optimizer = Adam(new_model.parameters(), lr=0.001)
    new_fsdp_model, new_fsdp_optimizer = load_fabric.setup(new_model, new_optimizer)
    
    new_fsdp_state = {"model": new_fsdp_model, "optimizer": new_fsdp_optimizer}
    metadata = checkpointer.load(checkpoint_path, new_fsdp_state, load_type="sharded")
    
    # Verify metadata
    assert metadata["epoch"] == 9
    assert metadata["step"] == 900
    
    # Verify parameters match
    loaded_state = new_fsdp_model.state_dict()
    success = all(torch.allclose(original_state[k], loaded_state[k]) for k in original_state)
    assert success, "FSDP Fabric sharded parameter verification failed"
    
    logger.info("✅ Test 9: FSDP sharded save/load with Fabric passed!")
    return True


def main():
    """Run all tests"""
    # Check test mode from environment variable
    test_mode = os.environ.get("FSDP_TEST_MODE", "single_process")
    
    # Only run tests on rank 0 when using torchrun to avoid duplicate output
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    
    if rank == 0:
        if test_mode == "multi_process" and world_size > 1:
            logger.info(f"🎯 Starting Checkpointer Multi-Process Tests (world_size={world_size})")
        elif test_mode == "single_process":
            logger.info("🎯 Starting Checkpointer Single-Process Tests")
        else:
            logger.info("🎯 Starting Checkpointer Tests")
    
    runner = TestRunner()
    
    # Select tests based on mode
    all_tests = [
        test_1_baseline_model_save_load,
        test_2_fsdp_save_normal_load_no_fabric,
        test_3_non_fsdp_partial_save_load_no_fabric,
        test_4_fsdp_partial_save_normal_load_no_fabric,
        test_5_fsdp_save_normal_load_with_fabric,
        test_6_non_fsdp_partial_save_load_with_fabric,
        test_7_fsdp_partial_fsdp_save_normal_load_with_fabric,
        test_8_fsdp_sharded_save_load,
        test_9_fsdp_fabric_sharded_save_load,
    ]
    
    try:
        if len(sys.argv) > 1:
            # Run specific test by index
            test_index = int(sys.argv[1])
            if 0 <= test_index < len(all_tests):
                all_tests[test_index](runner)
            else:
                logger.error(f"Invalid test index {test_index}. Valid range: 0-{len(all_tests)-1}")
                return 1
        else:
            # Run all tests
            for i, test_func in enumerate(all_tests):
                logger.info(f"Running test {i}: {test_func.__name__}")
                test_func(runner)
    finally:
        runner.teardown()
    
    return 0  # Non-rank-0 processes return success


if __name__ == "__main__":
    sys.exit(main()) 