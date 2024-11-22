import dataclasses
from typing import Dict, Any
import pandas as pd
import torch
import os
import uuid

LOG_INTERVAL = 1
WANDB_INTERVAL = 50
WANDB_WATCH_INTERVAL = 1000

@dataclasses.dataclass
class LoggerArgs:
    log_interval: int = LOG_INTERVAL
    wandb_interval: int = WANDB_INTERVAL
    wandb_watch_interval: int = WANDB_WATCH_INTERVAL
    experiment_name: str = "experiment"


class Logger:
    def __init__(self, 
                 log_interval=LOG_INTERVAL, 
                 wandb_interval=WANDB_INTERVAL, 
                 wandb_watch_interval=WANDB_WATCH_INTERVAL, 
                 experiment_name="experiment",
                 run_id: str = None,
                 hyper_params_dict: Dict[str, Any] = None,
                 wandb_run: "wandb.wandb_run.Run" = None):
        """
        use_wandb : True / False / wandb module
        """
        self.metrics = []
        self.n_iter = 0
        self.log_interval = log_interval
        self.wandb_interval = wandb_interval
        self.wandb_watch_interval = wandb_watch_interval
        self.wandb_run = wandb_run
        self.run_id = run_id if run_id is not None else str(uuid.uuid4())
        self.hyper_params_dict = hyper_params_dict if hyper_params_dict is not None else {}
        self.experiment_name = experiment_name
        if self.wandb_run:
            assert self.experiment_name == wandb_run.project

    @classmethod
    def from_args(cls, logger_args: LoggerArgs, hparam_or_hparam_list, use_wandb=True):
        # use_wandb : True / False / wandb run
        hyper_params_dict = {}
        if isinstance(hparam_or_hparam_list, list):
            for i in hparam_or_hparam_list:
                hyper_params_dict.update(i.__dict__)
        else:
            hyper_params_dict = hparam_or_hparam_list.__dict__

        if use_wandb == True:
            import wandb
            run = wandb.init(project=logger_args.experiment_name, config=hyper_params_dict)
        elif use_wandb == False:
            run = None
        else:
            assert isinstance(use_wandb, wandb.wandb_run.Run)
            assert use_wandb.project == logger_args.experiment_name
            run = use_wandb
        run_id = run.id if run else None
        return Logger(**logger_args.__dict__, run_id=run_id, hyper_params_dict=hyper_params_dict, wandb_run=run)

    def log_hists(self, model: torch.nn.Module, x, y):
        import wandb
        if self.wandb_run is None:
            return
        if self.n_iter % self.wandb_watch_interval != 0:
            return
        def hist(name, data):
            self.wandb_run.log({f"{name}_log2": wandb.Histogram(torch.log(data.detach().cpu()))}, step=self.n_iter)
        def act_hook(module, input, output):
            hist(f"act/{module.name}", output)
        def back_hook(module, input, output):
            hist(f"act_grad/{module.name}", output[0])

        for name, module in model.named_modules():
            module.name = f"{name}.{module.__class__.__name__}"

        with torch.nn.modules.module.register_module_forward_hook(act_hook):
            with torch.nn.modules.module.register_module_full_backward_hook(back_hook):
                loss, _ = model.loss_acc(x, y).values()
                loss.backward()
        
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                hist(f"weight/{module.name}", module.weight)
                if module.weight.grad is not None:
                    hist(f"weight_grad/{module.name}", module.weight.grad)

    def log(self, metrics: Dict[str, float]) -> None:
        if self.log_interval != -1 and self.n_iter % self.log_interval == 0:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics[key] = value
            self.metrics.append(metrics)
        if self.wandb_run and self.wandb_interval != -1 and self.n_iter % self.wandb_interval == 0:
            self.wandb_run.log(metrics, step=self.n_iter)
        self.n_iter += 1

    def log_same_iter(self, metrics: Dict[str, float]) -> None:
        try:
            previous_metrics: dict = self.metrics[-1]
        except:
            return self.log(metrics)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[key] = value
        previous_metrics.update(metrics)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.n_iter)

    def to_df(self) -> pd.DataFrame:
        try: 
            return pd.DataFrame(self.metrics)
        except:
            print("Cannot convert to DataFrame")
            return pd.DataFrame()

    def save_experiment(self, directory: str = "experiment_metrics"):
        """Saves the experiment configuration and metrics.

        Args:
            directory (str): The directory to save experiment data.
        """
        os.makedirs(os.path.join(directory, self.experiment_name), exist_ok=True)

        # Save hyper-parameters
        config_path = os.path.join(directory, self.experiment_name,"runs_summary.csv")
        config = self.hyper_params_dict.copy()
        config["run_id"] = self.run_id
        try:
            if not os.path.exists(config_path):
                pd.DataFrame([config]).to_csv(config_path, index=False)
            else:
                pd.DataFrame([config]).to_csv(config_path, mode='a', index=False)
        except Exception as e:
            print(f"Failed to save experiment configuration: {e}")

        # Save run-specific metrics
        run_path = os.path.join(directory, self.experiment_name,f"run_{self.run_id}.csv")
        try:
            self.to_df().to_csv(run_path, index=False)
            print(f"Run metrics saved to {run_path}")
        except Exception as e:
            print(f"Failed to save run metrics: {e}")

    def reset(self) -> None:
        self.metrics = []
        self.n_iter = 0

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __contains__(self, key):
        return key in self.metrics

    def __len__(self):
        return len(self.metrics)


class ExperimentMetrics:
    def __init__(self, directory: str):
        """
        Initialize the ExperimentLoader.

        Args:
            directory (str): The directory where experiment data is stored.
        """
        self.directory = directory
        self.runs_config = None
        self.runs: Dict[str, pd.DataFrame] = {}

        # Load experiment configuration upon initialization
        self._load_runs_config()

    def _load_runs_config(self):
        """
        Load the experiment configuration from the runs_summary.csv file.
        """
        config_path = os.path.join(self.directory, "runs_summary.csv")
        if os.path.exists(config_path):
            try:
                self.runs_config = pd.read_csv(config_path)
            except Exception as e:
                print(f"Failed to load experiment configuration: {e}")
        else:
            print("Experiment configuration file not found.")

    def get_runs_config(self):
        """
        Get the experiment configuration.

        Returns:
            pd.DataFrame: The experiment configuration.
        """
        return self.runs_config.copy()

    def save(self, directory, experiment_name):
        """
        Save the experiment configuration and metrics to a directory.

        Args:
            directory (str): The directory to save the experiment data.
        """
        os.makedirs(os.path.join(directory, experiment_name), exist_ok=True)
        self.runs_config.to_csv(os.path.join(directory, experiment_name, "runs_summary.csv"), index=False)
        for run_id in self.runs:
            self.runs[run_id].to_csv(os.path.join(directory, experiment_name,f"run_{run_id}.csv"), index=False)

    def get_run(self, run_id: str):
        """
        Get the metrics for a specific run, loading on demand if necessary.

        Args:
            run_id (str): The ID of the run.

        Returns:
            pd.DataFrame: The metrics for the specified run, or None if not found.
        """
        if run_id in self.runs:
            return self.runs[run_id].copy()

        file_path = os.path.join(self.directory, f"run_{run_id}.csv")
        if os.path.exists(file_path):
            try:
                run = pd.read_csv(file_path)
                self.runs[run_id] = run
                return run.copy()
            except Exception as e:
                print(f"Failed to load run data for {run_id}: {e}")
                return None
        else:
            print(f"Run file for {run_id} not found.")
            return None

    def run_ids(self):
        """
        List all run IDs available in the experiment from the experiment configuration.

        Returns:
            List[str]: A list of run IDs.
        """
        if self.runs_config is not None:
            return self.runs_config["run_id"].unique().tolist()
        else:
            print("Experiment configuration not loaded.")
            return []

    
