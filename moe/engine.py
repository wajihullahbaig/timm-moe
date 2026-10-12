"""Optimizer setup, one training epoch, evaluation."""
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .losses import LOSSES

BACKBONE_LR, HEAD_LR = 1e-4, 2e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Tracks the best validation score and signals when it has not improved for `patience` epochs."""

    def __init__(self, patience):
        self.patience, self.best, self.bad_epochs = patience, float('-inf'), 0

    def update(self, score):
        """Records `score`; returns True if it is a new best."""
        improved = score > self.best
        self.best, self.bad_epochs = (score, 0) if improved else (self.best, self.bad_epochs + 1)
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def build_optimizer(model, steps):
    """AdamW with a lower lr for the pretrained backbone, plus a one-cycle schedule over `steps`:
    warmup to the peak lr over the first 20% of steps, then cosine decay to ~0."""
    head = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    groups = [{'params': model.backbone.parameters(), 'lr': BACKBONE_LR}, {'params': head, 'lr': HEAD_LR}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[BACKBONE_LR, HEAD_LR], total_steps=steps,
        pct_start=0.2, anneal_strategy='cos', final_div_factor=1e4,
    )
    return optimizer, scheduler


def compute_losses(out, labels, loss_weights):
    """Returns (total, per-term detached values) for the task loss plus the weighted auxiliary losses."""
    terms = {'task': F.cross_entropy(out.logits.float(), labels)}
    terms.update({name: LOSSES[name](out, labels) for name in loss_weights})
    total = terms['task'] + sum(loss_weights[name] * terms[name] for name in loss_weights)
    return total, {name: value.detach() for name, value in terms.items()}


def train_epoch(model, loader, optimizer, scheduler, scaler, loss_weights, amp):
    """Trains one epoch; returns (mean loss terms, train accuracy %). Metrics stay on the GPU until the end."""
    model.train()
    sums, correct = defaultdict(float), 0
    for inputs, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out = model(inputs)
        loss, terms = compute_losses(out, targets, loss_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for name, value in terms.items():
            sums[name] += value
        correct += (out.logits.argmax(1) == targets).sum()
    means = {name: float(value) / len(loader) for name, value in sums.items()}
    return means, 100.0 * float(correct) / (len(loader) * loader.batch_size)


@torch.no_grad()
def evaluate(model, loader, amp):
    """Returns (accuracy %, usage) where usage[class, expert] is the mean gate weight for that class."""
    model.eval()
    correct, usage, counts = 0, 0.0, 0.0
    for inputs, targets in loader:
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out = model(inputs)
        correct += (out.logits.argmax(1) == targets).sum()
        onehot = F.one_hot(targets, out.logits.size(1)).float()
        usage = usage + onehot.t() @ out.gate_weights
        counts = counts + onehot.sum(0)
    return 100.0 * float(correct) / len(loader.labels), (usage / counts[:, None]).cpu().numpy()
