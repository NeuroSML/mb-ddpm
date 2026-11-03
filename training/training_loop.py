from tqdm import tqdm
from functools import partial

import torch
import torch.nn.functional as F

from runners.conditional_multi_bridge import q_sample as cmb_q_sample

from runners.ddpm import compute_label as ddpm_compute_label
from runners.conditional_bridge import compute_label as cb_compute_label
from runners.conditional_multi_bridge import compute_label as cmb_compute_label

def run_training_loop(config, diffusion, net, ema, optimizer, sched, sampler, loader, device, distributed=False):

    # sampling and diffusion functions
    if config["method"] == "conditional":
        compute_label = partial(cmb_compute_label, compute_label_fn=cb_compute_label)
        xT_in_batch = True
    elif config["method"] == "marginal":
        compute_label = partial(cmb_compute_label, compute_label_fn=ddpm_compute_label)
        xT_in_batch = False
        

    # training loop
    net.train()
    
    loss_history = []
    N_epoch = config["training"]["N_epoch"]
    print(f"Training model for {N_epoch} epochs...", flush=True)
    
    for epoch in range(N_epoch):

        if distributed:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{N_epoch}]", ncols=100)
        
        # number of samples
        print("number of samples", len(loader))

        for idx, data in enumerate(loop):
            
            # ===== sample boundary pair =====
            data = data.to(device)
            x0 = data.x0
            xT = data.xT
            t = data.t.squeeze()
            step = (t * config["diffusion"]["interval"]).floor().to(torch.long)
            mask = step != config["diffusion"]["interval"] - 1
            
            # ===== compute loss =====
            xt = cmb_q_sample(diffusion=diffusion, step=step, x0=x0, xT=xT)
            label = compute_label(
                diffusion=diffusion, step=step, x0=x0, xt=xt, xT=xT
            )

            if xT_in_batch:
                x_batch = torch.cat([xt, xT], dim=1)
            else:
                x_batch = xt

            pred = net(x_batch, step)
            assert xt.shape == label.shape == pred.shape

            loss = F.mse_loss(pred[mask], label[mask], reduction="none")
            loss = loss.mean()

            if torch.isnan(loss):
                print("NaN loss encountered, skipping this batch.")
                continue

            # backprop and optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ema.update()
            epoch_loss += loss.item()
            num_batches += 1

        if sched is not None:
            sched.step()

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{N_epoch}], " f"Loss: {avg_loss:.4f}")


    return net
