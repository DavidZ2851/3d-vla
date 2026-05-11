from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, ConstantLR

from .tristage_scheduler import TriStageLRScheduler


def fetch_scheduler(type_, optimizer, train_iters):
    if type_ == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=train_iters)
    elif type_ == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_iters)
    elif type_ == "hold_decay":
        hold_steps = train_iters*0.5
        decay_steps = train_iters*0.5
        scheduler_hold = ConstantLR(optimizer, factor=1.0, total_iters=hold_steps)
        scheduler_decay = CosineAnnealingLR(optimizer, T_max=decay_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_hold, scheduler_decay],
            milestones=[hold_steps]
        )
    elif type_ == "tristage_flower":
        base_lrs = [group.get("lr", 0.0) for group in optimizer.param_groups]
        peak_lr = max(base_lrs) if base_lrs else 0.0
        scheduler = TriStageLRScheduler(
            optimizer,
            total_steps=train_iters,
            lr=peak_lr,
        )
    else:
        raise NotImplementedError

    return scheduler
