def get_config_training(
    N_epoch,
    batch_size,
    lr,
    l2_norm,
    num_workers,
    method,
    data_dir,
    model_conditional_path,
    model_marginal_path,
    source_name,
    target_name,
    img_channels,
    model_channels,
    num_res_blocks,
    attention_resolutions,
    channel_mult,
    ema,
    use_checkpoint,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    interval,
    beta_max,
    snr_threshold,
):
    
    source_name = source_name.split(",")
    n_sources = len(source_name)
    if method == "marginal":
        in_channels = n_sources * img_channels
        model_path = model_marginal_path
    elif method == "conditional":
        in_channels = 2 * n_sources * img_channels
        model_path = model_conditional_path
    else:
        raise ValueError("Invalid method. Must be either 'marginal' or 'conditional'.")
    out_channels = n_sources * img_channels

    return {
        "model": {
            "model_channels": model_channels,
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": [int(x) for x in attention_resolutions.split(",")],
            "in_channels": in_channels,
            "out_channels": out_channels,
            "channel_mult": tuple(map(int, channel_mult.split(","))),
            "use_checkpoint": use_checkpoint,
            "num_heads": num_heads,
            "num_head_channels": num_head_channels,
            "num_heads_upsample": num_heads_upsample,
            "use_scale_shift_norm": use_scale_shift_norm,
            "dropout": dropout,
            "resblock_updown": resblock_updown,
            "use_fp16": use_fp16,
            "use_new_attention_order": use_new_attention_order,
            "model_path": model_path
        },
        "diffusion": {
            "interval": interval,
            "beta_max": beta_max,
            "nfe": None,
            "snr_threshold": snr_threshold,
            "n_timestep": interval,
            "linear_end": beta_max / interval,
            "schedule": "linear",
            "n_sources": n_sources,
                },
        "data": {
            "data_dir": data_dir,
            "source_name": source_name,
            "target_name": target_name,
                },
        "training": {
            "N_epoch": N_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "l2_norm": l2_norm,
            "num_workers": num_workers,
            "ema": ema,
        },
        "method": method,
    }


def get_config_sample(
    data_dir,
    model_conditional_path,
    model_marginal_path,
    source_name,
    target_name,
    img_channels,
    model_channels,
    num_res_blocks,
    attention_resolutions,
    channel_mult,
    ema,
    use_checkpoint,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    interval,
    beta_max,
    snr_threshold,
):
    source_name = source_name.split(",")
    n_sources = len(source_name)
    in_channels_marginal = n_sources * img_channels
    in_channels_conditional = 2 * n_sources * img_channels
    out_channels = n_sources * img_channels

    model_config = {
        "model_channels": model_channels,
        "num_res_blocks": num_res_blocks,
        "attention_resolutions": [int(x) for x in attention_resolutions.split(",")],
        "out_channels": out_channels,
        "channel_mult": tuple(map(int, channel_mult.split(","))),
        "use_checkpoint": use_checkpoint,
        "num_heads": num_heads,
        "num_head_channels": num_head_channels,
        "num_heads_upsample": num_heads_upsample,
        "use_scale_shift_norm": use_scale_shift_norm,
        "dropout": dropout,
        "resblock_updown": resblock_updown,
        "use_fp16": use_fp16,
        "use_new_attention_order": use_new_attention_order,
    }

    return {
        "diffusion": {
            "interval": interval,
            "beta_max": beta_max,
            "nfe": None,
            "snr_threshold": snr_threshold,
            "n_timestep": interval,
            "linear_end": beta_max / interval,
            "schedule": "linear",
            "n_sources": n_sources,
        },
        "model_conditional": {
            **model_config,
            "model_path": model_conditional_path,
            "in_channels": in_channels_conditional,
        },
        "model_marginal": {
            **model_config,
            "model_path": model_marginal_path,
            "in_channels": in_channels_marginal,
        },
        "data": {
            "data_dir": data_dir,
            "source_name": source_name,
            "target_name": target_name,
        },
        "training": {
            "ema": ema,
        },
    }

