import torch

from runners.ddpm import compute_pred_x0 as ddpm_compute_pred_x0
from runners.conditional_bridge import compute_pred_x0 as cb_compute_pred_x0
from runners.conditional_multi_bridge import compute_pred_x0 as cmb_compute_pred_x0

from runners.conditional_multi_bridge import p_posterior as cmb_p_posterior

@torch.no_grad()
def pred_x0_fn(
    diffusion, step, xt, xT, out_marginal, out_conditional, clip_denoise
):
    """ hat{x}_0^{hybrid}(x^{1:K}_t, x^{1:K}_T, t) = 
    (1 - lambda_t) hat{x}_0^{marg}(x^{1:K}_t, t) + 
    lambda_t hat{x}_0^{cond}(x^{1:K}_t, x^{1:K}_T, t) """

    # indicator for terminal step (t = T)
    terminal_step = step == diffusion.max_t

    # use hat{x}_0^{cond}(x^{1:K}_T, x^{1:K}_T, t) where xT approx x_{T-1}
    if terminal_step:
        step -= 1

    # compute signal to noise weight
    lambda_t = diffusion.lambd[step]

    # marginal estimator hat{x}_0^{marg}(x^{1:K}_t, t) = E[x0|xt^1:K]
    pred_x0_marginal = cmb_compute_pred_x0(
        diffusion=diffusion,
        step=step,
        xt=xt,
        net_out=out_marginal,
        compute_pred_x0_fn=ddpm_compute_pred_x0,
    )

    # conditional estimator hat{x}_0^{cond}(x^{1:K}_t, x^{1:K}_T, t) = E[x0|xt^1:K, xT^1:K]
    pred_x0_conditional = cmb_compute_pred_x0(
        diffusion=diffusion,
        step=step,
        xt=xt,
        xT=xT,
        net_out=out_conditional,
        compute_pred_x0_fn=cb_compute_pred_x0,
    )

    # hybrid estimator hat{x}_0^{hybrid}(x^{1:K}_t, x^{1:K}_T, t)
    pred_x0 = lambda_t * pred_x0_conditional + (1 - lambda_t) * pred_x0_marginal

    # clip denoised prediction to valid range
    if clip_denoise:
        n_channels = pred_x0.shape[1]
        if n_channels == 1:
            pred_x0.clamp_(-1.0, 1.0)
        elif n_channels == 3:
            pred_x0.clamp_(0, 1.0)

    return pred_x0


@torch.no_grad()
def p_posterior_fn(diffusion, prev_step, step, xt, x0, sigma_rec=torch.tensor(1e-3)):

    # indicator for first step (t = 0)
    first_step = prev_step == 0

    if first_step:
        return x0 + sigma_rec * torch.randn_like(x0)
    else:
        return cmb_p_posterior(
            diffusion,
            prev_step,
            step,
            xt=xt,
            x0=x0,
        )


@torch.no_grad()
def run_ddpm_sampling(
    xT,
    steps,
    diffusion,
    net_marginal,
    net_conditional,
    clip_denoise,
):

    # determine at which steps to log intermediate predictions
    log_steps_size = 200
    log_steps = steps
    log_steps = list(range(log_steps_size, log_steps[-1] + 1, log_steps_size))

    # create pairs of adjacent steps for sampling loop
    steps_rev = steps[::-1]
    pair_steps = zip(steps_rev[1:], steps_rev[:-1])

    # initialize xt^{1:K} = xT^{1:K} at time t = T
    xt = xT.clone()

    # save xt^{1:K} at time T
    progress = []
    progress.append(xt)

    # run sampling loop
    for prev_step, step in pair_steps:
        assert prev_step < step, f"{prev_step=}, {step=}"
        
        # create a tensor of t of shape (batch_size,) 
        step_tensor = torch.full(
            (xt.shape[0],), step, device=xt.device, dtype=torch.long
        )
        
        # concatenate xt^{1:K} and xT^{1:K} along channel dimension 
        xt_large = torch.cat([xt, xT], dim=1)

        # compute model outputs
        out_marginal = net_marginal(xt, step_tensor)
        out_conditional = net_conditional(xt_large, step_tensor)

        # compute the hybrid estimator hat{x}_0^hybrid
        pred_x0 = pred_x0_fn(
            diffusion, step, xt, xT, out_marginal, out_conditional, clip_denoise
        )

        # sample from the posterior distribution
        # x^{1:K}_{t-1} ~ p(x^{1:K}_{t-1} | x^{1:K}_t, x0 = pred_x0)
        xt = p_posterior_fn(diffusion, prev_step, step, xt, pred_x0)

        # keep track of x^{1:K}_t at log steps
        if prev_step in log_steps:
            progress.append(xt)

    # stack across log steps
    progress = torch.stack(progress, dim=0) 
    
    # progress was recorded for last step (t = T) and log_steps thereafter
    log_steps =  log_steps + [steps_rev[0]]
    log_steps = log_steps[::-1]
    
    return xt, progress, log_steps


