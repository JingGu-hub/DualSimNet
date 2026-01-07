import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import zipfile
import glob

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, SequentialLR


def zip_python_code(output_filename):
    """
    Zips all .py files in the current repository and saves it to the
    specified output filename.

    Args:
        output_filename: The name of the output zip file.
                         Defaults to "python_code_backup.zip".
    """

    with zipfile.ZipFile(output_filename, 'w') as zipf:
        files = glob.glob('models/**/*.py', recursive=True) + glob.glob('utils/**/*.py', recursive=True) + glob.glob('tasks/**/*.py', recursive=True) + glob.glob('*.py', recursive=True)
        for file in files:
            root = '/'.join(file.split('/')[:-1])
            nm = file.split('/')[-1]
            zipf.write(os.path.join(root, nm))

def ctm_mse_loss(predictions, certainties, inputs, use_most_certain=True):
    inputs_expanded = torch.repeat_interleave(inputs.unsqueeze(-1), predictions.size(-1), -1)
    # Losses are of shape [B, internal_ticks]
    losses = nn.MSELoss(reduction='none')(predictions, inputs_expanded)
    losses = losses.mean(-2).mean(-2)  # Average over ticks and classes

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:  # Revert to final loss if set
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2
    return loss, loss_index_2


class warmup():
    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps

    def step(self, current_step):
        if current_step < self.warmup_steps:  # current_step / warmup_steps * base_lr
            return float(current_step / self.warmup_steps)
        else:  # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return 1.0


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.00001,
            eta_min: float = 0.00001,
            last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                最適化手法インスタンス
            warmup_epochs (int):
                linear warmupを行うepoch数
            max_epochs (int):
                cosine曲線の終了に用いる 学習のepoch数
            warmup_start_lr (float):
                linear warmup 0 epoch目の学習率
            eta_min (float):
                cosine曲線の下限
            last_epoch (int):
                cosine曲線の位相オフセット
        学習率をmax_epochsに至るまでコサイン曲線に沿ってスケジュールする
        epoch 0からwarmup_epochsまでの学習曲線は線形warmupがかかる
        https://pytorch-lightning-bolts.readthedocs.io/en/stable/schedulers/warmup_cosine_annealing.html
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        return None

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (
                            1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class WarmupMultiStepLR(object):
    def __init__(self, optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.gamma = gamma

        # Define the warmup scheduler
        lambda_func = lambda step: step / warmup_steps if step < warmup_steps else 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda_func, last_epoch=last_epoch)

        # Define the multi-step scheduler
        multistep_scheduler = MultiStepLR(optimizer, milestones=[m - warmup_steps for m in milestones], gamma=gamma,
                                          last_epoch=last_epoch)

        # Chain the schedulers
        self.scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, multistep_scheduler],
                                      milestones=[warmup_steps])

    def step(self, epoch=None):
        self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the
    final dimension.

    Args:
      logits: A PyTorch tensor of logits.

    Returns:
      A PyTorch tensor containing the normalized entropy values.
    """

    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy
