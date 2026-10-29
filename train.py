import argparse
import torch
import yaml
import os 

import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch_ema import ExponentialMovingAverage

from config import get_config_training

from UNet.huggingface_unet import UNetModel  

from data.dataset import MultiBridgeDataset
from data.dataloaders import collate_fn

from diffusion import Diffusion, make_beta_schedule

from training.training_loop import run_training_loop
from training.utils import count_parameters, build_optimizer_sched

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--diffusion_config", type=str)
    parser.add_argument("--training_config", type=str)
    parser.add_argument("--method", type=str, default="conditional")
    args = parser.parse_args()

    # Device
    def get_torch_device():
        """Return 'cuda' if a GPU is available, else 'cpu'."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DEVICE = get_torch_device()
    print("Using device:", DEVICE)
    num_gpus = torch.cuda.device_count()
    distributed=num_gpus > 1
    print(f"Number of GPUs available: {num_gpus}")
    
    if distributed:
        local_rank = setup()
        DEVICE = torch.device(f"cuda:{local_rank}")
        print(f"Running on GPU {local_rank}")
        print(f"World size: {dist.get_world_size()}")
  
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    data_config = load_yaml(args.data_config)
    training_config = load_yaml(args.training_config)
    config = get_config_training(
        **{**model_config, **diffusion_config, **data_config, **training_config, 
        **{"method" : args.method}}
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
    net = UNetModel(**config["model"]).to(DEVICE)
    
    # move model to gpu and parallelize if multiple gpus available
    if distributed:
        net = DDP(net, device_ids=[local_rank])

    # Dataloaders
    dataset = MultiBridgeDataset(**config["data"])
    
    # sampler
    if distributed:
        sampler = DistributedSampler(dataset)
        batch_size = config["training"]["batch_size"] // dist.get_world_size()
    else:
        sampler = None
        batch_size = config["training"]["batch_size"]
        
    # loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if config["training"]["num_workers"] > 0 else False,
        sampler=sampler,
    )

    # count parameters 
    n_params = count_parameters(model=net)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Model with {n_params / (10**6)}M parameters", flush=True)

    # optimizer and scheduler
    params = list(net.parameters())
    ema = ExponentialMovingAverage(params, decay=config["training"]["ema"])
    optimizer, sched = build_optimizer_sched(config['training'], params)
    
    # run training loop
    net = run_training_loop(
        config=config,
        diffusion=diffusion,
        net=net,
        ema=ema,
        optimizer=optimizer,
        sched=sched,
        sampler = sampler,
        loader = loader,
        device=DEVICE,
        distributed=distributed,
    )
    
    # Save parameters
    model_params_file = config["model"]["model_path"]
    os.makedirs(os.path.dirname(model_params_file), exist_ok=True)
    print(f"Saving model parameters to {model_params_file}...")
    if distributed:
        torch.save(net.module.state_dict(), model_params_file)
    else:
        torch.save(net.state_dict(), model_params_file)


    

if __name__ == "__main__":
    main()