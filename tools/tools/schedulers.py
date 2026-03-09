
import math
import torch
import logging
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR, _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(level=logging.INFO)

class OneCycleLRContinueFlat(OneCycleLR):
    """
    Custom learning rate scheduler extending OneCycleLR. Continues with a flat learning rate after the one cycle schedule.

    Args:
        Refer to OneCycleLR for detailed arguments.

    Methods:
        get_lr(): Computes the learning rate at each step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return super().get_lr()


class CosineAnnealingLRContinueFlat(CosineAnnealingLR):
    """
    Custom learning rate scheduler extending CosineAnnealingLR. Continues with a flat learning rate after the cosine annealing schedule.

    Args:
        Refer to CosineAnnealingLR for detailed arguments.

    Methods:
        get_lr(): Computes the learning rate at each step.
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            return (self.eta_min if self.eta_min==0 
                    else [group['lr'] for group in self.optimizer.param_groups])
        else:
            return super().get_lr()

def get_scheduler(sch_type:str, optimizer, attr:dict):

    if sch_type.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **attr)
    elif sch_type.lower() == "constant":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif sch_type.lower() == "cycliclr":
        """
            torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, 
            step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, 
            scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
            last_epoch=- 1, verbose=False)
        """
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **attr)
    elif sch_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **attr)
    elif sch_type.lower().replace("_", "") == "singlecosine":
        scheduler = CosineAnnealingLRContinueFlat(optimizer, **attr)
    elif sch_type.lower() == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    **attr,
                )
    elif sch_type.lower() == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    **attr,
                )
    elif "warmup_to_constant" in sch_type.lower():
        scheduler = WarmupToConstant(optimizer, **attr)
    elif "OneCycleLR".lower() in sch_type.lower():
        scheduler = OneCycleLRContinueFlat(optimizer, **attr)
    elif "linear_warmup_cosine_decay".lower() in sch_type.lower():
        scheduler = linear_warmup_cosine_decay(optimizer, **attr)
    else:
        logging.warning("Scheduler unknown!")
        logging.WARNING("Returning None")
        scheduler = None
    
    return scheduler

class WarmupToConstant(_LRScheduler):
    """Gradually warm-up learning rate in optimizer to a constant value."""

    def __init__(self, optimizer: Optimizer, num_steps: int = 100):
        """
        args:
            optimizer (Optimizer): Wrapped optimizer.
            num_steps: target learning rate is reached at num_steps.
        """
        self.num_steps = num_steps
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.num_steps:
            return [base_lr for base_lr in self.base_lrs]
        return [
            (base_lr / self.num_steps) * self.last_epoch for base_lr in self.base_lrs
        ]


def plot_lr_sch(scheduler, optimizer, iterations = 100, plot=True):

    lr = []
    lr.append(optimizer.param_groups[0]['lr'])
    for i in range(iterations):
        scheduler.step()
        lr.append(optimizer.param_groups[0]['lr'])

    if plot:
        plt.plot(lr)
        plt.xlabel("Iterations")
        plt.ylabel("Learning rate")
    
    return lr

try:
    from pytorch_lightning import LightningModule
    def linear_warmup_cosine_decay(
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 100,
        final_factor: float = 1e-3,
        init_factor: float = 1e-1,
        warmup_ratio: float  = None,
        model: LightningModule=None,
    ) -> LambdaLR:
        """Return a scheduler with a linear warmup and a cosine decay."""
        # Replace the total_steps with the model trainer's actual max_steps
        
        total_steps = total_steps if total_steps is not None else get_max_steps(model)

        # Replace the wamup_steps with the ratio
        if warmup_ratio is not None:
            warmup_steps = int(warmup_ratio * total_steps)

        # Define the actual scheduler function
        def fn(x: int) -> float:
            if x < warmup_steps:
                return init_factor + x * (1 - init_factor) / max(1, warmup_steps)
            progress = (x - warmup_steps) / max(1, total_steps - warmup_steps)
            lr = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(final_factor, lr)

        # The lambda scheduler is the easiest way to define a custom scheduler
        return LambdaLR(optimizer, fn)



    if __name__ == "__main__":
        network = torch.nn.Linear(1,1)

        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-10)

        iterations = 200_000

        scheduler = get_scheduler("OneCycleLR", optimizer,
                                {"max_lr": 0.0005, "total_steps": 200_000,
                                "div_factor":100, "pct_start": 0.2}
                                    )

        lr = plot_lr_sch(scheduler, optimizer, iterations = iterations, plot=True)
        print(f"First 10 lr {lr[:10]}")
except:
    logging.warning("Pytorch Lightning not installed. Skipping linear_warmup_cosine_decay scheduler.")
