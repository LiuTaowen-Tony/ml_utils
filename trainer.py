import os
import abc
import collections
import dataclasses
import os
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Literal, Optional, Union, cast
import subprocess
import datetime

import torch
from lightning_utilities import apply_to_collection
from tqdm import tqdm
from torchmetrics import MeanMetric

import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects, is_wrapped
from transformers import get_cosine_schedule_with_warmup
import torch.utils.data
from lightning.fabric.strategies import FSDPStrategy
import wandb

# Import consolidated GitTagger
from .git_utils import GitTagger

# A FEW CHANGES TO LIGHTNING
# 1. always expect to see
#    a step level validation and checkpointing
# 2. scheduler need to be called at step level
# 3. need to use evaluator
# 4. lightning module need to define get_checkpointable_part

# need to implement 
# 1. on validatoin epoch start  requires dataloader_idx
# need to implement get_checkpointable_part
# need to implement get_optimizer


def write_postfix(
    prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
):
    if isinstance(prog_bar, tqdm) and candidates is not None:
        postfix_str = ""
        float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
        if isinstance(candidates, torch.Tensor):
            postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
        elif isinstance(candidates, Mapping):
            for k, v in float_candidates.items():
                postfix_str += f" {prefix}_{k}: {v:.3f}"

        if postfix_str:
            prog_bar.set_postfix_str(postfix_str)


class Algorithm(abc.ABC, torch.nn.Module):
    # define how to compute loss
    # define which part is updatable
    def __init__(self):
        super().__init__()
        self.wandb = None

    @abc.abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Any:
        pass
    
    @abc.abstractmethod
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int) -> Any:
        pass

    @abc.abstractmethod
    def get_updatable_part(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_start(self, dataloader_idx: int):
        pass
    
    def on_validation_epoch_end(self, dataloader_idx: int):
        pass

class OptimizerInterface(abc.ABC):
    @abc.abstractmethod
    def get_optimizer_scheduler(self, algorithm: Algorithm):
        pass

@dataclasses.dataclass
class DefaultOptimizer(OptimizerInterface):
    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float = 1e-5

    def get_optimizer_scheduler(self, algorithm: Algorithm):
        updatable_part = algorithm.get_updatable_part()
        optimizer = torch.optim.AdamW(
            updatable_part.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
            num_cycles=0.5,
        )
        return optimizer, scheduler

@dataclasses.dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    max_steps: Optional[int] = 10000
    grad_accum_steps: int = 1
    warmup_steps: float = 0.1
    validation_interval: int = 1000
    checkpoint_interval: Optional[int] = None
    grad_clip_value: float = 1.0
    weight_decay: float = 1e-5

    def __post_init__(self):
        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.validation_interval
        if self.warmup_steps < 1:
            self.warmup_steps = int(self.max_steps * self.warmup_steps)




class ModelRunner:
    def __init__(
        self,
        strategy: str | Strategy,
        train_config: TrainConfig,
        precision: str = "fp32",
        seed: int = 42,
        optimizer_manager: Optional[DefaultOptimizer] = None,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        train_from_previous_checkpoint: bool = False,
        git_tagger: Optional[GitTagger] = None,
        auto_tag_experiments: bool = True,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readability but with reduced
        featureset. As a trainer with more included features, we recommend using the
        :class:`lightning.pytorch.Trainer`.

        Args:
            strategy: Strategy for how to run across multiple devices. Possible choices are:
                ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"``, ``"fsdp"``.
            grad_clip_value: Gradient clipping value. negative for no clipping.
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.

        Warning:
            callbacks written for the lightning trainer (especially making assumptions on the trainer), won't work!

        """
        self.fabric = L.Fabric(
            strategy=strategy,
            precision=precision,
        )
        self.fabric.seed_everything(seed)
        torch.set_float32_matmul_precision('medium')

        self.train_config = train_config

        # will be saved when checkpointing
        self.optimize_step = 0
        self.current_epoch = 0
        self.batch_idx_in_current_epoch = 0


        self.use_distributed_sampler = use_distributed_sampler
        self.checkpoint_dir = checkpoint_dir

        if optimizer_manager is None:
            optimizer_manager = DefaultOptimizer(
                learning_rate=train_config.learning_rate, 
                max_steps=train_config.max_steps,
                warmup_steps=train_config.warmup_steps,
                weight_decay=train_config.weight_decay,
            )
        self.optimizer_manager = optimizer_manager
        assert train_config.max_steps == self.optimizer_manager.max_steps
        self.train_from_previous_checkpoint = train_from_previous_checkpoint

        # Git tagging setup
        self.auto_tag_experiments = auto_tag_experiments
        if git_tagger is None and auto_tag_experiments:
            git_tagger = GitTagger(
                tag_prefix="exp",
                max_tags_to_keep=50,  # Reasonable default for training
                strict_temp_check=False,  # Don't be too strict during training
                auto_stash_temp_files=True,  # Auto-handle temp files
            )

        self.git_tagger = git_tagger
        self.fabric.launch()
        
    def init_wandb(self, experiment_config: dict) -> "wandb.Run":
        if self.fabric.is_global_zero:
            name = experiment_config.get("name", None)
            return  wandb.init(
                project=experiment_config["project_name"],
                entity=experiment_config["entity"],
                config=experiment_config,
                name=name,
            )
        else:
            class _Dummy:  # makes .log() a no‑op on workers
                def log(self, *_, **__): ...
            return _Dummy()

    def ensure_wrap_dataloader(self, dataloader: torch.utils.data.DataLoader):
        if is_wrapped(dataloader):
            return dataloader
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=self.use_distributed_sampler)

    def ensure_wrap_algorightm(self, algorithm: Algorithm):
        if is_wrapped(algorithm):
            return algorithm
        return self.fabric.setup(algorithm)

    def ensure_wrap_optimizer(self, optimizer: torch.optim.Optimizer):
        if is_wrapped(optimizer):
            return optimizer
        return self.fabric.setup_optimizers(optimizer)

    def eval(self, algorithm: Algorithm, dataloader: torch.utils.data.DataLoader, dataloader_idx: int = 0):
        algorithm = self.ensure_wrap_algorightm(algorithm)
        dataloader = self.ensure_wrap_dataloader(dataloader)
        algorithm.eval()
        metrics_dict_recorder = MetricDictRecorder()
        self._on_validation_epoch_start(algorithm, dataloader_idx)
        for batch_idx, batch in enumerate(dataloader):
            result = algorithm.validation_step(batch, batch_idx, dataloader_idx)
            metrics_dict_recorder.update(result)
        self._on_validation_epoch_end(algorithm, dataloader_idx)
        metrics_dict = metrics_dict_recorder.compute()
        algorithm.train()
        return metrics_dict

    def _on_validation_epoch_start(self, algorithm: Algorithm, dataloader_idx: int):
        try:
            algorithm.on_validation_epoch_start(dataloader_idx)
        except:
            algorithm.on_validation_epoch_start()

    def _on_validation_epoch_end(self, algorithm: Algorithm, dataloader_idx: int):
        try:
            algorithm.on_validation_epoch_end(dataloader_idx)
        except:
            algorithm.on_validation_epoch_end()

    def model_report(self, algorithm: Algorithm):
        total_params = 0
        trainable_params = 0
        for param in algorithm.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return {
            "total_params(M)": total_params / 1e6,
            "trainable_params(M)": trainable_params / 1e6,
        }

    def run(
        self,
        algorithm: Algorithm,
        hyper_params: dict,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        val_loaders: list[torch.utils.data.DataLoader] = [],
        ckpt_path: Optional[str] = None,
    ):
        algorithm = self.ensure_wrap_algorightm(algorithm)
        optimizer, scheduler = self.optimizer_manager.get_optimizer_scheduler(algorithm)
        optimizer = self.ensure_wrap_optimizer(optimizer)

        # wrap dataloader and model
        val_loaders = [self.ensure_wrap_dataloader(loader) for loader in val_loaders]
        train_loader = self.ensure_wrap_dataloader(train_loader)

        # checkpoint
        # TODO: enable checkpoint from wandb run name
        checkpoint_model = algorithm.get_updatable_part()
        state = {"model": checkpoint_model, "optim": optimizer, "scheduler": scheduler}
        if ckpt_path is not None and os.path.isdir(ckpt_path) and self.train_from_previous_checkpoint:
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)

        self.wandb = self.init_wandb(hyper_params)
        algorithm.wandb = self.wandb
        
        # Create git tag for this experiment run and link to wandb
        if self.auto_tag_experiments and self.git_tagger is not None:
            experiment_name = hyper_params["project_name"] + "-" + self.wandb.name
            # Create git tag with minimal info (wandb has the config)
            tag_name = self.git_tagger.create_experiment_tag(
                experiment_name=experiment_name,
                experiment_config=None,  # wandb stores this
                push_to_remote=False
            )
            
            git_info = {
                "git_tag": tag_name,
                "git_commit": self.git_tagger._get_current_commit_hash(short=False),
                "git_branch": self.git_tagger._run_git_command(["git", "branch", "--show-current"]),
                "git_dirty": bool(self.git_tagger._get_uncommitted_files()["regular"])
            }
            self.wandb.log(git_info)
            # Also add to wandb config for easy access
            if hasattr(self.wandb, 'config'):
                self.wandb.config.update(git_info)
        
        algorithm.on_train_epoch_start()
        self.loss_weight_sum = 0.0
        if self.fabric.is_global_zero:
            model_report = self.model_report(algorithm)
            self.wandb.log(model_report)
            print(model_report)

        while True:
            # train loop
            iterable = train_loader
            if self.fabric.is_global_zero:
                iterable = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {self.current_epoch}")
            for batch_idx, batch in enumerate(iterable):
                # TODO: this is wrong, if we do distributed batching
                # we need to count the number of data points
                # because local batch size is divived by world size
                if self.batch_idx_in_current_epoch >= batch_idx:
                    continue # skip to batch_idx > self.batch_idx_in_current_epoch
                self.dataloader_step(batch, batch_idx, optimizer, scheduler, algorithm, val_loaders)
                self.batch_idx_in_current_epoch += 1
                self.optimize_step += 1
                if self.optimize_step >= self.train_config.max_steps:
                    self.save(state)
                    return
            self.current_epoch += 1
            self.batch_idx_in_current_epoch = 0

    def dataloader_step(
        self, 
        batch, 
        batch_idx: int, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler.LRScheduler, 
        algorithm: Algorithm, 
        val_loaders: list[torch.utils.data.DataLoader],
    ):
        # optimizer step
        if self.optimize_step % self.train_config.grad_accum_steps == 0:
            # update optimizer
            if self.optimize_step > 0:
                params = optimizer.param_groups[0]["params"]
                for param in params:
                    if param.grad is not None:
                        param.grad.data.div_(self.loss_weight_sum)
                # gradient clipping
                if self.train_config.grad_clip_value > 0:
                    self.fabric.clip_gradients(algorithm, optimizer, max_norm=self.train_config.grad_clip_value)
                
                self.loss_weight_sum = 0.0
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.optimize_step += 1
            
            # validation & checkpoint
            if self.optimize_step % self.train_config.validation_interval == 0:
                algorithm.on_train_epoch_end()
                for dataloader_idx, val_loader in enumerate(val_loaders):
                    metrics_dict = self.eval(algorithm, val_loader, dataloader_idx)
                    self.wandb.log(metrics_dict)
                algorithm.on_train_epoch_start()
            if self.optimize_step % self.train_config.checkpoint_interval == 0:
                checkpoint_model = algorithm.get_updatable_part()
                state = {"model": checkpoint_model, "optim": optimizer, "scheduler": scheduler}
                self.save(state)

        # train step
        loss_pack = algorithm.training_step(batch, batch_idx)
        loss, loss_weight = loss_pack, 1.0
        if isinstance(loss_pack, Mapping):
            loss_weight = loss_pack.get("loss_weight", 1.0)
            loss = loss_pack["loss"]
        loss = loss * loss_weight
        self.loss_weight_sum += float(loss_weight)
        self.fabric.backward(loss)




    def load(self, state: Mapping, path: str) -> None:
        try:
            # First try fabric.load (for FSDP checkpoints)
            remainder = self.fabric.load(path, state)
            self.optimize_step = remainder.pop("optimize_step")
            self.current_epoch = remainder.pop("current_epoch")
            self.batch_idx_in_current_epoch = remainder.pop("batch_idx_in_current_epoch")
        except Exception as e:
            checkpoint = torch.load(path, map_location="cpu")
            
            state["model"].load_state_dict(checkpoint["model"])
            state["optim"].load_state_dict(checkpoint["optim"])
            state["scheduler"].load_state_dict(checkpoint["scheduler"])
            
            self.optimize_step = checkpoint["optimize_step"]
            self.current_epoch = checkpoint["current_epoch"]
            self.batch_idx_in_current_epoch = checkpoint["batch_idx_in_current_epoch"]

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        if state is None:
            state = {}

        state.update(optimize_step=self.optimize_step, current_epoch=self.current_epoch, batch_idx_in_current_epoch=self.batch_idx_in_current_epoch)
        
        # Check if we have an FSDP strategy and non-FSDP model
        if isinstance(self.fabric.strategy, FSDPStrategy) and "model" in state:
            model = state["model"]
            # Check if model is FSDP-wrapped by looking for FSDP-specific attributes
            if not hasattr(model, '_fsdp_wrapped_module') and not any(hasattr(p, '_fsdp_wrapped_module') for p in model.parameters()):
                # Model is not FSDP-wrapped, save manually
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt")
                
                # Save non-FSDP model state manually
                model_state = model.state_dict() if hasattr(model, 'state_dict') else model
                save_state = {
                    "model": model_state,
                    "optimize_step": self.optimize_step,
                    "current_epoch": self.current_epoch,
                    "batch_idx_in_current_epoch": self.batch_idx_in_current_epoch,
                    "optim": state["optim"].state_dict(),
                    "scheduler": state["scheduler"].state_dict(),
                }
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                if self.fabric.is_global_zero:
                    torch.save(save_state, checkpoint_path)
                return
                
        self.fabric.save(os.path.join(self.checkpoint_dir, f".ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])


