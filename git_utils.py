"""
Git utilities for experiment tracking and versioning.
"""

import os
import subprocess
import datetime
import json
import hashlib
import re
import fnmatch
from typing import Optional, Dict, Any, List, Set
from pathlib import Path


class GitTagger:
    """Enhanced git tagger with experiment metadata support and temporary file protection."""
    
    # Common temporary file patterns to watch out for
    DEFAULT_TEMP_PATTERNS = [
        # Python temporary files
        "*.pyc", "**/__pycache__/**", "*.pyo", "*.pyd",
        # Jupyter notebooks checkpoints
        "**/.ipynb_checkpoints/**",
        # Editor temporary files
        "*~", "*.swp", "*.swo", ".*.swp", ".*.swo",
        # OS temporary files
        ".DS_Store", "Thumbs.db", "*.tmp", "*.temp",
        # Log files
        "*.log", "logs/**", "wandb/**",
        # Model checkpoints (often large temporary files)
        "checkpoints/**", "*.ckpt", "*.pth", "*.safetensors",
        # Data cache
        "**/.cache/**", "cache/**", "*.cache",
        # Build artifacts
        "build/**", "dist/**", "*.egg-info/**",
        # Environment files
        ".env", "*.env",
    ]
    
    def __init__(self, 
                 tag_prefix: str = "exp", 
                 include_timestamp: bool = True,
                 include_commit_hash: bool = True,
                 max_tags_to_keep: Optional[int] = 100,
                 metadata_file: str = ".experiment_metadata.json",
                 temp_file_patterns: Optional[List[str]] = None,
                 strict_temp_check: bool = True,
                 auto_stash_temp_files: bool = False,
                 require_clean_repo: bool = False):
        self.tag_prefix = tag_prefix
        self.include_timestamp = include_timestamp
        self.include_commit_hash = include_commit_hash
        self.max_tags_to_keep = max_tags_to_keep
        self.metadata_file = metadata_file
        
        # Temporary file handling
        self.temp_file_patterns = temp_file_patterns or self.DEFAULT_TEMP_PATTERNS
        self.strict_temp_check = strict_temp_check
        self.auto_stash_temp_files = auto_stash_temp_files
        self.require_clean_repo = require_clean_repo
        
    def _run_git_command(self, cmd: list[str]) -> Optional[str]:
        """Run a git command and return the output."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                cwd=os.getcwd()
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return None
    
    def _get_current_commit_hash(self, short: bool = True) -> Optional[str]:
        """Get the current git commit hash."""
        cmd = ["git", "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        return self._run_git_command(cmd)
    
    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        return self._run_git_command(["git", "rev-parse", "--git-dir"]) is not None
    
    def _is_temp_file(self, file_path: str) -> bool:
        """Check if a file matches temporary file patterns."""
        for pattern in self.temp_file_patterns:
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(os.path.basename(file_path), pattern):
                return True
        return False
    
    def _get_uncommitted_files(self) -> Dict[str, List[str]]:
        """Get categorized uncommitted files (regular vs temporary)."""
        diff_output = self._run_git_command(["git", "status", "--porcelain"])
        
        if not diff_output:
            return {"regular": [], "temporary": [], "ignored": []}
        
        files = {"regular": [], "temporary": [], "ignored": []}
        
        for line in diff_output.split('\n'):
            if not line.strip():
                continue
                
            # Parse git status format: XY filename
            status = line[:2]
            filename = line[3:].strip()
            
            # Skip ignored files (shouldn't appear in --porcelain, but just in case)
            if status == '!!':
                files["ignored"].append(filename)
                continue
            
            # Check if it's a temporary file
            if self._is_temp_file(filename):
                files["temporary"].append(filename)
            else:
                files["regular"].append(filename)
        
        return files
    
    def _get_repo_status(self) -> Dict[str, Any]:
        """Get current repository status with temporary file analysis."""
        status = {}
        
        # Get branch name
        branch = self._run_git_command(["git", "branch", "--show-current"])
        status["branch"] = branch
        
        # Get commit hash
        commit_hash = self._get_current_commit_hash(short=False)
        status["commit_hash"] = commit_hash
        status["short_hash"] = self._get_current_commit_hash(short=True)
        
        # Get categorized uncommitted files
        file_status = self._get_uncommitted_files()
        status["has_uncommitted_changes"] = bool(file_status["regular"] or file_status["temporary"])
        status["uncommitted_files"] = file_status["regular"] + file_status["temporary"]
        status["regular_files"] = file_status["regular"]
        status["temporary_files"] = file_status["temporary"]
        status["ignored_files"] = file_status["ignored"]
        
        # Get remote URL
        remote_url = self._run_git_command(["git", "remote", "get-url", "origin"])
        status["remote_url"] = remote_url
        
        return status
    
    def _handle_temporary_files(self, repo_status: Dict[str, Any]) -> bool:
        """Handle temporary files based on configuration. Returns True if safe to proceed."""
        temp_files = repo_status.get("temporary_files", [])
        regular_files = repo_status.get("regular_files", [])
        
        # No uncommitted files - all good
        if not temp_files and not regular_files:
            return True
        
        # Handle temporary files
        if temp_files:
            print(f"⚠️  Found {len(temp_files)} temporary files:")
            for f in temp_files[:5]:  # Show first 5
                print(f"    {f}")
            if len(temp_files) > 5:
                print(f"    ... and {len(temp_files) - 5} more")
        
        # Handle regular uncommitted files
        if regular_files:
            print(f"📝 Found {len(regular_files)} uncommitted files:")
            for f in regular_files[:5]:  # Show first 5
                print(f"    {f}")
            if len(regular_files) > 5:
                print(f"    ... and {len(regular_files) - 5} more")
        
        # Apply policies
        if self.require_clean_repo and (temp_files or regular_files):
            print("❌ require_clean_repo=True but repository has uncommitted changes")
            return False
        
        if self.auto_stash_temp_files and temp_files:
            print("🗂️  Auto-stashing temporary files...")
            # Create a list of temp files for stashing
            temp_file_list = " ".join(f'"{f}"' for f in temp_files)
            stash_cmd = f"git stash push -m 'Auto-stash temp files for experiment tagging' {temp_file_list}"
            
            # Use shell=True for complex command with quotes
            try:
                subprocess.run(stash_cmd, shell=True, check=True, capture_output=True, text=True)
                print("✅ Temporary files stashed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to stash temporary files: {e.stderr}")
                if self.strict_temp_check:
                    return False
        
        if self.strict_temp_check and temp_files and not self.auto_stash_temp_files:
            print("❌ strict_temp_check=True but temporary files present")
            print("💡 Solutions:")
            print("   1. Set auto_stash_temp_files=True")
            print("   2. Set strict_temp_check=False") 
            print("   3. Clean up temporary files manually")
            return False
        
        return True
    
    def _save_gitignore_recommendations(self, temp_files: List[str]):
        """Save recommendations for .gitignore based on found temporary files."""
        if not temp_files:
            return
        
        gitignore_path = Path(".gitignore")
        
        # Read existing .gitignore
        existing_patterns = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing_patterns = {line.strip() for line in f if line.strip() and not line.startswith('#')}
        
        # Find new patterns to suggest
        suggested_patterns = set()
        for temp_file in temp_files:
            # Suggest directory patterns for files in directories
            if '/' in temp_file:
                dir_pattern = temp_file.split('/')[0] + '/**'
                if dir_pattern not in existing_patterns:
                    suggested_patterns.add(dir_pattern)
            
            # Suggest extension patterns
            if '.' in temp_file:
                ext_pattern = '*' + Path(temp_file).suffix
                if ext_pattern not in existing_patterns:
                    suggested_patterns.add(ext_pattern)
        
        if suggested_patterns:
            print(f"\n💡 Consider adding these patterns to .gitignore:")
            for pattern in sorted(suggested_patterns):
                print(f"    {pattern}")
    
    def _save_git_snapshot_metadata(self, tag_name: str, experiment_config: Optional[Dict[str, Any]] = None):
        """Save minimal git snapshot metadata (config should be in wandb)."""
        repo_status = self._get_repo_status()
        
        metadata = {
            "tag": tag_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit": repo_status["commit_hash"],
            "git_branch": repo_status["branch"],
            "git_dirty": bool(repo_status["regular_files"]),
            "temp_files_found": len(repo_status["temporary_files"]),
            "uncommitted_files": len(repo_status["regular_files"])
        }
        
        # Only save config if explicitly provided (usually wandb handles this)
        if experiment_config:
            metadata["experiment_config"] = experiment_config
            metadata["config_hash"] = hashlib.md5(json.dumps(experiment_config, sort_keys=True).encode()).hexdigest()
        
        # Load existing metadata
        metadata_path = Path(self.metadata_file)
        all_metadata = []
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    all_metadata = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                all_metadata = []
        
        # Add new metadata
        all_metadata.append(metadata)
        
        # Keep only recent entries
        if self.max_tags_to_keep:
            all_metadata = all_metadata[-self.max_tags_to_keep:]
        
        # Save back
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
    
    def _cleanup_old_tags(self):
        """Remove old experiment tags if max_tags_to_keep is set."""
        if self.max_tags_to_keep is None:
            return
            
        # Get all tags with our prefix, sorted by creation date
        tags_output = self._run_git_command([
            "git", "tag", "-l", f"{self.tag_prefix}-*", 
            "--sort=-creatordate"
        ])
        
        if not tags_output:
            return
            
        tags = tags_output.split('\n')
        if len(tags) > self.max_tags_to_keep:
            old_tags = tags[self.max_tags_to_keep:]
            for tag in old_tags:
                print(f"🗑️  Removing old tag: {tag}")
                self._run_git_command(["git", "tag", "-d", tag])
    
    def create_experiment_tag(self, 
                            experiment_name: Optional[str] = None,
                            experiment_config: Optional[Dict[str, Any]] = None,
                            push_to_remote: bool = False,
                            force: bool = False) -> Optional[str]:
        """Create a git tag for the current experiment run with temporary file protection."""
        if not self._is_git_repo():
            print("⚠️  Warning: Not in a git repository, skipping tag creation")
            return None
        
        # Check repository status
        repo_status = self._get_repo_status()
        
        # Handle temporary files
        if not force and not self._handle_temporary_files(repo_status):
            print("❌ Tagging aborted due to temporary file policy")
            print("💡 Use force=True to override, or clean up files first")
            return None
        
        # Provide .gitignore recommendations
        temp_files = repo_status.get("temporary_files", [])
        if temp_files:
            self._save_gitignore_recommendations(temp_files)
        
        # Build tag name
        tag_parts = [self.tag_prefix]
        
        if experiment_name:
            # Clean experiment name for git tag
            clean_name = "".join(c for c in experiment_name if c.isalnum() or c in '-_').lower()
            tag_parts.append(clean_name)
            
        if self.include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tag_parts.append(timestamp)
            
        if self.include_commit_hash:
            commit_hash = self._get_current_commit_hash()
            if commit_hash:
                tag_parts.append(commit_hash)
        
        tag_name = "-".join(tag_parts)
        
        # Create git tag message focused on reproducibility
        tag_message = f"Experiment: {experiment_name or 'unnamed'}\n"
        tag_message += f"Timestamp: {datetime.datetime.now().isoformat()}\n"
        tag_message += f"Branch: {repo_status['branch']}\n"
        tag_message += f"Commit: {repo_status['commit_hash']}\n"
        
        if repo_status.get('regular_files'):
            tag_message += f"⚠️ Uncommitted changes: {len(repo_status['regular_files'])} files\n"
        if repo_status.get('temporary_files'):
            tag_message += f"🗂️ Temp files handled: {len(repo_status['temporary_files'])} files\n"
        
        tag_message += "📊 Config stored in: wandb\n"  # Point to wandb for config
        
        # Create the tag
        result = self._run_git_command([
            "git", "tag", "-a", tag_name, 
            "-m", tag_message
        ])
        
        if result is not None:
            print(f"✅ Created git tag: {tag_name}")
            
            # Save minimal git snapshot metadata
            self._save_git_snapshot_metadata(tag_name, experiment_config)
            
            # Push to remote if requested
            if push_to_remote:
                push_result = self._run_git_command(["git", "push", "origin", tag_name])
                if push_result is not None:
                    print(f"✅ Pushed tag to remote: {tag_name}")
                else:
                    print(f"❌ Failed to push tag to remote: {tag_name}")
            
            self._cleanup_old_tags()
            return tag_name
        else:
            print(f"❌ Failed to create git tag: {tag_name}")
            return None
    
    def add_temp_pattern(self, pattern: str):
        """Add a new temporary file pattern."""
        if pattern not in self.temp_file_patterns:
            self.temp_file_patterns.append(pattern)
    
    def remove_temp_pattern(self, pattern: str):
        """Remove a temporary file pattern."""
        if pattern in self.temp_file_patterns:
            self.temp_file_patterns.remove(pattern)
    
    def check_temp_files(self) -> Dict[str, List[str]]:
        """Check for temporary files without creating a tag."""
        return self._get_uncommitted_files()
    
    def list_experiment_tags(self, limit: int = 10) -> list[str]:
        """List recent experiment tags."""
        tags_output = self._run_git_command([
            "git", "tag", "-l", f"{self.tag_prefix}-*", 
            "--sort=-creatordate"
        ])
        
        if not tags_output:
            return []
        
        tags = tags_output.split('\n')
        return tags[:limit]
    
    def get_experiment_metadata(self, tag_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific experiment tag."""
        if not Path(self.metadata_file).exists():
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                all_metadata = json.load(f)
            
            if tag_name is None:
                return all_metadata[-1] if all_metadata else None
            
            for metadata in reversed(all_metadata):
                if metadata.get("tag") == tag_name:
                    return metadata
            
            return None
        except (json.JSONDecodeError, FileNotFoundError):
            return None


def auto_tag_on_run(func):
    """Decorator to automatically create git tags when running experiments."""
    def wrapper(*args, **kwargs):
        tagger = GitTagger(
            strict_temp_check=True,  # Be strict by default
            auto_stash_temp_files=False  # Don't auto-stash by default
        )
        
        # Try to extract experiment name from function name or kwargs
        experiment_name = kwargs.get('experiment_name', func.__name__)
        
        # Create git snapshot (let wandb handle config)
        tag_name = tagger.create_experiment_tag(
            experiment_name=experiment_name,
            experiment_config=None  # Config should be in wandb/external tracking
        )
        
        # Add git info to kwargs for the function to use
        if tag_name:
            kwargs['_git_tag'] = tag_name
            kwargs['_git_commit'] = tagger._get_current_commit_hash(short=False)
            kwargs['_git_branch'] = tagger._run_git_command(["git", "branch", "--show-current"])
        
        try:
            result = func(*args, **kwargs)
            if tag_name:
                print(f"✅ Experiment '{experiment_name}' completed successfully")
                print(f"   📍 Git tag: {tag_name}")
                print(f"   💡 Use wandb or your tracking system for config storage")
            return result
        except Exception as e:
            if tag_name:
                print(f"❌ Experiment '{experiment_name}' failed (tag: {tag_name})")
                print(f"Error: {e}")
            raise
    
    return wrapper


if __name__ == "__main__":
    # Example usage with temporary file protection
    tagger = GitTagger(
        strict_temp_check=True,
        auto_stash_temp_files=True,
        temp_file_patterns=[
            "*.pyc", "**/__pycache__/**", 
            "*.log", "wandb/**", "checkpoints/**"
        ]
    )
    
    # Check for temporary files
    temp_status = tagger.check_temp_files()
    print("Temporary file check:", temp_status)
    
    experiment_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "llama-7b",
        "compression_method": "kv_compression"
    }
    
    tag_name = tagger.create_experiment_tag(
        experiment_name="kv_compression_test",
        experiment_config=experiment_config,
        push_to_remote=False
    )
    
    print(f"Created tag: {tag_name}")
    print("Recent tags:", tagger.list_experiment_tags()) 