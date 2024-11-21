from typing import Dict
import pandas as pd
import torch
import argparse


LOG_INTERVAL = 1
WANDB_INTERVAL = 50
WANDB_WATCH_INTERVAL = 1000

def add_log_args(parser : argparse.ArgumentParser, 
                 defualt_experiment_name : str = "trial",
                 default_log_interval : int = LOG_INTERVAL,
                 default_wandb_interval : int = WANDB_INTERVAL,
                 default_wandb_watch_interval : int = WANDB_WATCH_INTERVAL):
    parser.add_argument("--log_interval", type=int, default=default_log_interval)
    parser.add_argument("--wandb_interval", type=int, default=default_wandb_interval)
    parser.add_argument("--wandb_watch_interval", type=int, default=default_wandb_watch_interval)
    parser.add_argument("--experiment_name", type=str, default=defualt_experiment_name)
    return parser


class Logger:
    def __init__(self, 
                 log_interval=LOG_INTERVAL, 
                 wandb_interval=WANDB_INTERVAL, 
                 wandb_watch_interval=WANDB_WATCH_INTERVAL, 
                 wandb_run: "wandb.wandb_run.Run" =None):
        """
        use_wandb : True / False / wandb module
        """
        self.metrics = []
        self.n_iter = 0
        self.log_interval = log_interval
        self.wandb_interval = wandb_interval
        self.wandb_watch_interval = wandb_watch_interval
        self.wandb_run = wandb_run

    @classmethod
    def from_args(cls, args, use_wandb=True):
        # use_wandb : True / False / wandb run
        if use_wandb:
            import wandb
            run = wandb.init(project=args.experiment_name, config=args)
        return cls(args.log_interval, args.wandb_interval, args.wandb_watch_interval, run)


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
            previous_metrics:dict  = self.metrics[-1]
        except:
            print("no previous metrics")
            return
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
            print("Cannot convert to csv")
            return pd.DataFrame()

    def to_csv(self, path: str) -> None:
        self.to_df().to_csv(path)

    def reset(self) -> None:
        self.metrics = {}
        self.n_iter = 0

    def __getitem__(self, key):
        return self.metrics[key]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __contains__(self, key):
        return key in self.metrics

    def __len__(self):
        return len(self.metrics)