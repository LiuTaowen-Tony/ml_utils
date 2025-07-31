# Git Tagging for Experiment Tracking

This module provides automatic git tagging functionality to help track your machine learning experiments. Every time you run an experiment, a git tag is automatically created with experiment metadata, making it easy to reproduce results and track changes.

## Features

- 🏷️ **Automatic git tagging** when experiments start
- 📊 **Experiment metadata storage** (config, git status, timestamps)
- 🔍 **Easy experiment history browsing**
- 🧹 **Automatic cleanup** of old tags
- 🚀 **Multiple integration options** (decorator, manual, trainer integration)
- 📝 **Detailed tag messages** with commit info and config hashes
- 🛡️ **Temporary file protection** - prevents accidental inclusion of temp files
- 🗂️ **Auto-stashing** of temporary files
- 💡 **Smart .gitignore recommendations**

## Quick Start

### 1. Decorator Approach (Simplest)

```python
from ml_utils.git_utils import auto_tag_on_run

@auto_tag_on_run
def my_experiment(learning_rate=0.001, batch_size=32):
    # Your experiment code here
    return {"accuracy": 0.95}

# Automatically creates tag when called
result = my_experiment(learning_rate=0.0005, batch_size=64)
```

### 2. ModelRunner Integration (Recommended for Training)

```python
from ml_utils.trainer import ModelRunner, TrainConfig, GitTagger

# Configure trainer with automatic tagging
runner = ModelRunner(
    strategy="ddp",
    train_config=TrainConfig(learning_rate=0.001, max_steps=10000),
    auto_tag_experiments=True,  # Enable auto-tagging
    git_tagger=GitTagger(tag_prefix="my-exp")  # Optional custom tagger
)

# Tag automatically created when training starts
runner.run(algorithm, hyper_params, train_loader, val_loaders)
```

### 3. Manual Control (Most Flexible)

```python
from ml_utils.git_utils import GitTagger

tagger = GitTagger(
    tag_prefix="kv-compression",
    max_tags_to_keep=50,
    strict_temp_check=True,  # Be strict about temporary files
    auto_stash_temp_files=False  # Manual control over temp files
)

# Create tag with full metadata
tag_name = tagger.create_experiment_tag(
    experiment_name="zigzag_attention_v2",
    experiment_config={
        "learning_rate": 0.001,
        "block_size": 128,
        "model": "llama-7b"
    },
    push_to_remote=False
)
```

## Tag Naming Convention

Tags are automatically named with this format:
```
{prefix}-{experiment_name}-{timestamp}-{commit_hash}
```

Examples:
- `exp-kv_compression-20241201-143022-a1b2c3d`
- `trainer-exp-zigzag_attention-20241201-143545-x9y8z7w`

## Configuration Options

### GitTagger (Basic)
```python
GitTagger(
    tag_prefix="experiment",        # Prefix for all tags
    include_timestamp=True,         # Include timestamp in tag name
    include_commit_hash=True,       # Include git commit hash
    max_tags_to_keep=50            # Auto-cleanup old tags (None = no limit)
)
```

### GitTagger (Full-Featured)
```python
GitTagger(
    tag_prefix="exp",                        # Tag prefix
    include_timestamp=True,                  # Include timestamp
    include_commit_hash=True,                # Include commit hash
    max_tags_to_keep=100,                   # Max tags to keep
    metadata_file=".experiment_metadata.json", # Metadata storage file
    strict_temp_check=True,                  # Strict temporary file checking
    auto_stash_temp_files=False,             # Auto-stash temporary files
    require_clean_repo=False                 # Require completely clean repo
)
```

## Temporary File Protection

The git tagger automatically detects and handles temporary files to prevent them from being accidentally included in your experiment snapshots.

### Default Temporary File Patterns
The system automatically recognizes these common temporary files:
- **Python**: `*.pyc`, `**/__pycache__/**`, `*.pyo`, `*.pyd`
- **Jupyter**: `**/.ipynb_checkpoints/**`
- **Editors**: `*~`, `*.swp`, `*.swo`, `.*.swp`, `.*.swo`
- **OS**: `.DS_Store`, `Thumbs.db`, `*.tmp`, `*.temp`
- **Logs**: `*.log`, `logs/**`, `wandb/**`
- **Models**: `checkpoints/**`, `*.ckpt`, `*.pth`, `*.safetensors`
- **Cache**: `**/.cache/**`, `cache/**`, `*.cache`
- **Build**: `build/**`, `dist/**`, `*.egg-info/**`
- **Env**: `.env`, `*.env`

### Protection Modes

#### Strict Mode (Recommended for Production)
```python
tagger = GitTagger(
    strict_temp_check=True,      # Abort if temp files found
    auto_stash_temp_files=False  # Manual cleanup required
)
```

#### Permissive Mode (Good for Development)
```python
tagger = GitTagger(
    strict_temp_check=False,     # Allow temp files
    auto_stash_temp_files=True   # Auto-stash them
)
```

#### Ultra-Clean Mode (Maximum Safety)
```python
tagger = GitTagger(
    require_clean_repo=True      # Require zero uncommitted changes
)
```

### What Happens When Temp Files Are Found

**Strict Mode + No Auto-Stash:**
```
⚠️  Found 3 temporary files:
    __pycache__/model.cpython-39.pyc
    logs/training.log
    checkpoints/model_temp.ckpt
❌ strict_temp_check=True but temporary files present
💡 Solutions:
   1. Set auto_stash_temp_files=True
   2. Set strict_temp_check=False
   3. Clean up temporary files manually
```

**Auto-Stash Mode:**
```
⚠️  Found 3 temporary files:
    __pycache__/model.cpython-39.pyc
    logs/training.log
    checkpoints/model_temp.ckpt
🗂️  Auto-stashing temporary files...
✅ Temporary files stashed successfully
💡 Consider adding these patterns to .gitignore:
    __pycache__/**
    logs/**
    checkpoints/**
```

### Custom Temporary File Patterns
```python
tagger = GitTagger(
    temp_file_patterns=[
        "*.pyc", "**/__pycache__/**",     # Python
        "my_temp_dir/**",                  # Custom temp directory
        "*.backup",                        # Backup files
        "experiment_*.json"                # Temp experiment files
    ]
)

# Add patterns dynamically
tagger.add_temp_pattern("*.tmp_model")
tagger.remove_temp_pattern("*.log")  # Allow log files

# Check what would be considered temporary
temp_status = tagger.check_temp_files()
print("Temporary files:", temp_status["temporary"])
print("Regular files:", temp_status["regular"])
```

## Metadata Storage

The git tagger stores detailed metadata in JSON format:

```json
{
  "tag": "exp-kv_compression-20241201-143022-a1b2c3d",
  "timestamp": "2024-12-01T14:30:22.123456",
  "git_status": {
    "branch": "main",
    "commit_hash": "a1b2c3d4e5f6...",
    "short_hash": "a1b2c3d",
    "has_uncommitted_changes": false,
    "uncommitted_files": [],
    "remote_url": "git@github.com:user/repo.git"
  },
  "experiment_config": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": "llama-7b"
  },
  "config_hash": "d4c3b2a1e5f6..."
}
```

## Browsing Experiment History

### List Recent Tags
```python
tagger = GitTagger()

# Get 10 most recent experiment tags
recent_tags = tagger.list_experiment_tags(limit=10)
for tag in recent_tags:
    print(tag)
```

### Get Experiment Metadata
```python
# Get metadata for latest experiment
latest = tagger.get_experiment_metadata()

# Get metadata for specific tag
specific = tagger.get_experiment_metadata("exp-kv_compression-20241201-143022-a1b2c3d")

print(f"Experiment: {latest['experiment_config']}")
print(f"Git commit: {latest['git_status']['commit_hash']}")
print(f"Had uncommitted changes: {latest['git_status']['has_uncommitted_changes']}")
```

## Integration Examples

### With Weights & Biases
```python
# In ModelRunner, git tag is automatically logged to wandb
runner = ModelRunner(auto_tag_experiments=True)
hyper_params = {
    "project_name": "my_project",
    "experiment_name": "test_run"
}
runner.run(algorithm, hyper_params, train_loader, val_loaders)
# Wandb will show git_tag in metrics
```

### Custom Experiment Function
```python
def run_hyperparameter_sweep():
    tagger = GitTagger(
        tag_prefix="hp-sweep",
        strict_temp_check=False,  # Allow temp files during sweeps
        auto_stash_temp_files=True  # Auto-handle them
    )
    
    for lr in [0.001, 0.0005, 0.0001]:
        for bs in [32, 64, 128]:
            config = {"learning_rate": lr, "batch_size": bs}
            
            tag_name = tagger.create_experiment_tag(
                experiment_name=f"lr{lr}_bs{bs}",
                experiment_config=config
            )
            
            # Run experiment with this config
            run_training(config)
            print(f"Completed: {tag_name}")
```

## Best Practices

### 1. Use Descriptive Experiment Names
```python
# Good
tagger.create_experiment_tag("kv_compression_block128_lr001")

# Bad
tagger.create_experiment_tag("test1")
```

### 2. Include Full Configuration
```python
config = {
    "model": "llama-7b",
    "dataset": "openwebtext",
    "learning_rate": 0.001,
    "batch_size": 32,
    "block_size": 128,
    "compression_ratio": 0.5,
    "optimizer": "adamw",
    "weight_decay": 1e-5
}
tagger.create_experiment_tag("full_config_exp", config)
```

### 3. Set Reasonable Tag Limits
```python
# For active development
GitTagger(max_tags_to_keep=50)

# For long-term storage
GitTagger(max_tags_to_keep=200)

# No limit (be careful!)
GitTagger(max_tags_to_keep=None)
```

### 4. Use Different Prefixes for Different Experiment Types
```python
# For hyperparameter sweeps
hp_tagger = GitTagger(tag_prefix="hp-sweep")

# For ablation studies
ablation_tagger = GitTagger(tag_prefix="ablation")

# For baseline comparisons
baseline_tagger = GitTagger(tag_prefix="baseline")
```

## Troubleshooting

### Not in a Git Repository
```
Warning: Not in a git repository, skipping tag creation
```
**Solution**: Initialize git repository with `git init`

### Uncommitted Changes
```
Warning: Repository has uncommitted changes:
  M src/model.py
  M config.yaml
```
**Solution**: This is just a warning. Tags are still created, but metadata will show uncommitted changes.

### Tag Already Exists
```
Git command failed: git tag -a exp-test-20241201-143022-a1b2c3d -m ...
Error: tag 'exp-test-20241201-143022-a1b2c3d' already exists
```
**Solution**: This is rare due to timestamps, but you can delete the existing tag:
```bash
git tag -d exp-test-20241201-143022-a1b2c3d
```

### Permission Issues with Remote Push
```
❌ Failed to push tag to remote: exp-test-20241201-143022-a1b2c3d
```
**Solution**: Check git credentials and repository permissions.

## Command Line Usage

You can also run the example script directly:
```bash
python examples/git_tagging_example.py
```

Or use the utility module:
```bash
python -m ml_utils.git_utils
```

## Advanced Features

### Custom Tag Messages
The automatic tag messages include:
- Experiment name and timestamp
- Git branch and commit hash
- Configuration hash for reproducibility
- Uncommitted changes warning (if any)

### Automatic Cleanup
Old tags are automatically removed based on `max_tags_to_keep` setting, keeping your git repository clean while preserving recent experiment history.

### Integration with CI/CD
You can disable tagging in CI environments:
```python
import os
auto_tag = not os.getenv('CI', False)  # Disable in CI

runner = ModelRunner(auto_tag_experiments=auto_tag)
``` 