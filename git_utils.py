"""
Git utilities for experiment tracking and versioning.
"""

import os
import subprocess
import datetime
import fnmatch
from typing import Optional, List


class GitTagger:
    """Simple git tagger that creates experiment tags and ignores temporary files."""
    
    # Common temporary file patterns to ignore
    TEMP_PATTERNS = [
        "*.pyc", "__pycache__/**", "*.pyo", "*.pyd",
        "**/.ipynb_checkpoints/**",
        "*~", "*.swp", "*.swo", ".*.swp", ".*.swo",
        ".DS_Store", "Thumbs.db", "*.tmp", "*.temp",
        "*.log", "logs/**", "wandb/**", "lightning_logs/**",
        "checkpoints/**", "*.ckpt", "*.pth", "*.safetensors",
        "**/.cache/**", "cache/**", "*.cache",
        "build/**", "dist/**", "*.egg-info/**",
        ".env", "*.env",
    ]
    
    def __init__(self):
        pass
    
    def _is_rank_zero(self) -> bool:
        """Check if this is rank 0 in distributed training."""
        # Check common distributed training environment variables
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except ImportError:
            pass
        
        # Default to True if no distributed training detected
        return True
    
    def _run_git_command(self, cmd: List[str]) -> Optional[str]:
        """Run a git command and return the output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        return self._run_git_command(["git", "rev-parse", "--git-dir"]) is not None
    
    def _is_temp_file(self, file_path: str) -> bool:
        """Check if a file matches temporary file patterns."""
        for pattern in self.TEMP_PATTERNS:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        return False
    
    def _get_non_temp_files(self) -> List[str]:
        """Get list of uncommitted files that are NOT temporary."""
        status_output = self._run_git_command(["git", "status", "--porcelain"])
        if not status_output:
            return []
        
        non_temp_files = []
        for line in status_output.split('\n'):
            if not line.strip():
                continue
            filename = line[3:].strip()  # Skip status prefix
            if not self._is_temp_file(filename):
                non_temp_files.append(filename)
        
        return non_temp_files
    
    def create_tag(self, experiment_name: str, auto_commit: bool = True) -> Optional[str]:
        """Create a git tag for experiment, optionally committing changes first."""
        # Only create tags on rank 0 in distributed training
        if not self._is_rank_zero():
            return None
            
        if not self._is_git_repo():
            print("⚠️  Not in a git repository, skipping tag creation")
            return None
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # Create tag name with timestamp
        tag_name = f"{experiment_name}-{timestamp}"

        # Check for non-temporary uncommitted files
        non_temp_files = self._get_non_temp_files()
        
        if non_temp_files and auto_commit:
            print(f"📝 Auto-committing {len(non_temp_files)} files (ignoring temp files):")
            for f in non_temp_files[:3]:
                print(f"    {f}")
            if len(non_temp_files) > 3:
                print(f"    ... and {len(non_temp_files) - 3} more")
            
            # Add only non-temp files
            for file in non_temp_files:
                self._run_git_command(["git", "add", file])
            
            # Create commit
            if self._run_git_command(["git", "commit", "-m", tag_name]):
                print(f"✅ Created commit: {tag_name}")
            else:
                print("❌ Failed to create commit")
                return None
                
        elif non_temp_files:
            print(f"⚠️  Warning: {len(non_temp_files)} uncommitted files (ignoring temp files):")
            for f in non_temp_files[:3]:
                print(f"    {f}")
            if len(non_temp_files) > 3:
                print(f"    ... and {len(non_temp_files) - 3} more")
            print("💡 Use auto_commit=True to commit automatically")
        
        
        # Create the tag
        if self._run_git_command(["git", "tag", "-a", tag_name, "-m", tag_name]):
            print(f"✅ Created git tag: {tag_name}")
            return tag_name
        else:
            print(f"❌ Failed to create git tag")
            return None
    
    def list_tags(self, limit: int = 10) -> List[str]:
        """List recent experiment tags."""
        tags_output = self._run_git_command([
            "git", "tag", "-l", 
            "--sort=-creatordate"
        ])
        
        if not tags_output:
            return []
        
        tags = tags_output.split('\n')
        return tags[:limit]


def git_tag(experiment_name: str, auto_commit: bool = True) -> Optional[str]:
    """Quick function to create an experiment tag with optional auto-commit."""
    tagger = GitTagger()
    return tagger.create_tag(experiment_name, auto_commit=auto_commit)


if __name__ == "__main__":
    # Simple usage examples
    tagger = GitTagger()
    
    # Create tag without committing
    tag_name = tagger.create_tag("my_experiment")
    
    # Create tag with auto-commit
    tag_name = tagger.create_tag("my_experiment", auto_commit=True)
    
    if tag_name:
        print(f"Created tag: {tag_name}")
        print("Recent tags:", tagger.list_tags()) 