import argparse
import os
import torch
import yaml
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader

from config import get_config_sample

from UNet.huggingface_unet import UNetModel  

from data.dataset import MultiBridgeDataset
from data.dataloaders import collate_fn

from diffusion import Diffusion, make_beta_schedule

from sampling.ddpm_sampling import run_ddpm_sampling
from sampling.util import space_indices

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--diffusion_config", type=str)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    print("save_dir: ", args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'target'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'source'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'progress'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'recon'), exist_ok=True)

    # Device
    def get_torch_device():
        """Return 'cuda' if a GPU is available, else 'cpu'."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEVICE = get_torch_device()
    print("Using device:", DEVICE)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    data_config = load_yaml(args.data_config)
    config = get_config_sample(
        **{**model_config, **diffusion_config, **data_config}
    )
    print("Loaded configuration:", config)

    # seed
    set_seed(42)

    # Diffusion process (symmetric beta schedule)
    betas = make_beta_schedule(**config["diffusion"])
    betas = np.concatenate(
        [
            betas[: diffusion_config["interval"] // 2],
            np.flip(betas[: diffusion_config["interval"] // 2]),
        ]
    )
    diffusion = Diffusion(
        betas=betas, n_sources=config["diffusion"]["n_sources"], 
        snr_threshold=config["diffusion"]["snr_threshold"], device=DEVICE
    )

    # conditional diffusion net
    net_conditional = UNetModel(**config["model_conditional"]).to(DEVICE)
    model_conditional_path = config["model_conditional"]["model_path"]
    paths_exist = os.path.exists(model_conditional_path)
    if paths_exist:
        print(
            f"Model parameters file found at {model_conditional_path}. Loading model parameters..."
        )

        state_dict = torch.load(
            model_conditional_path,
            map_location=DEVICE,
        )

        net_conditional.load_state_dict(state_dict)

        ema_params = net_conditional.parameters()
        ema_conditional = ExponentialMovingAverage(
            ema_params,
            decay=config["training"]["ema"],
        )

    else:
        raise FileNotFoundError(
            f"Model parameters file not found at {model_conditional_path}. Please check the path."
        )

    # marginal diffusion net
    net_marginal = UNetModel(**config["model_marginal"]).to(DEVICE)
    model_marginal_path = config["model_marginal"]["model_path"]
    paths_exist = os.path.exists(model_marginal_path)
    if paths_exist:
        print(
            f"Model parameters file found at {model_marginal_path}. Loading model parameters..."
        )

        state_dict = torch.load(
            model_marginal_path,
            map_location=DEVICE,
        )

        net_marginal.load_state_dict(state_dict)

        ema_params = net_marginal.parameters()
        ema_marginal = ExponentialMovingAverage(
            ema_params,
            decay=config["training"]["ema"],
        )

    else:
        raise FileNotFoundError(
            f"Model parameters file not found at {model_marginal_path}. Please check the path."
        )

    # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
    # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
    # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
    nfe = config["diffusion"]["nfe"]
    nfe = nfe or config["diffusion"]["interval"] - 1
    assert 0 < nfe < config["diffusion"]["interval"] == len(diffusion.betas)
    steps = space_indices(config["diffusion"]["interval"], nfe + 1)

    # Load diffusion sampler
    sample_fn = partial(
        run_ddpm_sampling,
        net_marginal=net_marginal,
        net_conditional=net_conditional,
        diffusion=diffusion,
        clip_denoise=True,
        steps=steps,
    )

    # Dataloaders
    dataset = MultiBridgeDataset(**config["data"])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    with ema_marginal.average_parameters(), ema_conditional.average_parameters():
        device = next(net_conditional.parameters()).device
        net_marginal.eval()
        net_conditional.eval()

        for i, batch in enumerate(loader):
            print(f"Processing batch {i+1}/{len(loader)}...")
            
            batch = batch.to(device)
            x0 = batch.x0
            xT = batch.xT
            x0_hat, progress, log_steps = sample_fn(
                xT.clone(),
            )
            
            # x0: target image (batch_size, img_channels, H, W)
            # xT: source images (batch_size, n_sources * img_channels, H, W)
            # x0_hat: predicted target image (batch_size, img_channels, H, W)
            # progress: list of predicted target image at each log step (len(log_steps), batch_size, img_channels, H, W)
            
            # save as image 
            fname = batch.sample_id[0] + '.png'
            
            # "target": x0
            target_arr = x0[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, n_sources * img_channels)
            plt.imsave(os.path.join(args.save_dir, 'target', fname), target_arr)
            
            # source_k: xT^k
            img_channels = xT.shape[1] // config["diffusion"]["n_sources"]
            for k in range(config["diffusion"]["n_sources"]):
                source_arr = xT[0, k * img_channels : (k + 1) * img_channels].cpu().numpy().transpose(1, 2, 0)
                plt.imsave(os.path.join(args.save_dir, 'source', f"{batch.sample_id[0]}_source_{k}.png"), source_arr)
            
            # source_k_step_t : xt^k
            progress = progress.clamp_(0, 1.0)
            for step_idx, step in enumerate(log_steps):
                for k in range(config["diffusion"]["n_sources"]):
                    progress_arr = progress[step_idx, 0, k * img_channels : (k + 1) * img_channels].cpu().numpy().transpose(1, 2, 0)
                    plt.imsave(os.path.join(args.save_dir, 'progress', f"{batch.sample_id[0]}_source_{k}_step_{step}.png"), progress_arr)
                    
            # target_hat: x0_hat 
            target_hat_arr = x0_hat[0].clamp_(0, 1.0).cpu().numpy().transpose(1, 2, 0)
            plt.imsave(os.path.join(args.save_dir, 'recon', fname), target_hat_arr)
                     

    print("Sampling completed. Results saved in:", args.save_dir)

if __name__ == "__main__":
    main()
