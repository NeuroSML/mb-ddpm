import torch
from diffusion import compute_gaussian_product_coef


def compute_label(diffusion, step, x0, xt, **kwargs):
    """(xt - x0) / std_fwd
    assuming xt = x0 + epsilon*std_fwd"""
    coef_fwd, std_fwd = diffusion.get_param_fwd(step, xdim=x0.shape[1:])
    label = (xt - coef_fwd * x0) / std_fwd
    return label.detach()


def compute_pred_x0(diffusion, step, xt, net_out, **kwargs):
    """E[x0| xt] = (xt - std_fwd * net_out) / coef_fwd"""
    coef_fwd, std_fwd = diffusion.get_param_fwd(step, xdim=xt.shape[1:])
    return (xt - std_fwd * net_out) / coef_fwd


def p_posterior(diffusion, prev_step, step, xt, x0):
    """sample x_{t-1} ~ p(x_{t-1} | xt, x_0)"""

    assert prev_step < step

    # p(x_{t-1} | x0) = Normal(x_{t-1}, coef_prev_step*x0, std_prev_step**2)
    coef_prev_step = diffusion.coef_fwd[prev_step]
    std_prev_step = diffusion.std_fwd[prev_step]

    # p(x_{t-1} | xt) propto exp(-1/(2*std_bwdstep**2) (x_{t-1} - coef_bwdstep*xt)^2)
    coef_bwdstep = 1 / diffusion.coef_onestep[step]
    std_bwdstep = diffusion.std_onestep[step] / diffusion.coef_onestep[step]

    # p(x_{t-1} | xt, x_0) = Normal(x0*mu_bs_1 + xt*mu_bs_2, var)
    mu_bs_1, mu_bs_2, var = compute_gaussian_product_coef(
        coef_prev_step, coef_bwdstep, std_prev_step, std_bwdstep
    )
    mu = mu_bs_1 * x0 + mu_bs_2 * xt

    xt_prev = mu
    if prev_step > 0:
        xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

    return xt_prev
