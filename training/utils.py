from torch.optim import AdamW, lr_scheduler


def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return n_params


def build_optimizer_sched(config, params):

    optim_dict = {"lr": config["lr"], "weight_decay": config["l2_norm"]}
    optimizer = AdamW(params, **optim_dict)
    sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["N_epoch"])

    return optimizer, sched
