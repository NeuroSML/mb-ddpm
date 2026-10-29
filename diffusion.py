from abc import ABC
import os
import torch
import numpy as np
from functools import partial

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def make_beta_schedule(
    schedule="linear",
    n_timestep=1000,
    linear_start=1e-4,
    linear_end=2e-2,
    cosine_s=8e-3,
    **kwargs,
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def compute_gaussian_product_coef(coef1, coef2, sigma1, sigma2):
    """Given p1 = N(y|coef1*x_1, sigma_1**2) and p2 = N(y|coef2*x_2, sigma_2**2)
    return p1 * p2 = N(y| mu_1 * x1 + mu_2 * x2, var)"""

    denom = sigma1**2 + sigma2**2
    mu_1 = (sigma2**2 * coef1) / denom
    mu_2 = (sigma1**2 * coef2) / denom
    var = (sigma1**2 * sigma2**2) / denom
    return mu_1, mu_2, var


class Diffusion(ABC):
    def __init__(self, betas, n_sources, snr_threshold, device):

        self.device = device
        self.max_t = len(betas) - 1
        self.n_sources = n_sources

        # tensorize betas
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)

        # alpha, alpha_bar, alpha_tilde
        alphas = 1.0 - self.betas
        alphabar = torch.cumprod(alphas, dim=0)
        alphatilde = alphabar[-1] / alphabar

        # one step: p(xt | x_{t-1}) = Normal(xt; coef_onestep * x_{t-1}, std_onestep^2)
        self.coef_onestep = torch.sqrt(alphas)
        self.std_onestep = torch.sqrt(1 - alphas)

        # forward: p(xt | x0) = Normal(xt; coef_fwd * x0, std_fwd^2)
        self.coef_fwd = torch.sqrt(alphabar)
        self.std_fwd = torch.sqrt(1.0 - alphabar)

        # forward target: p(xT | xt) = Normal(xT; coef_fwdtarget * xt, std_fwdtarget^2)
        self.coef_fwdtarget = torch.sqrt(alphatilde)
        self.std_fwdtarget = torch.sqrt((1.0 - alphatilde))

        # backward: p(xt | xT) propto exp(- 1/(2*std_bwd^2) ||xt - coef_bwd*xT||^2)
        self.coef_bwd = 1 / self.coef_fwdtarget
        self.std_bwd = self.std_fwdtarget / self.coef_fwdtarget

        # forward conditional: p(xt | x0, xT) = Normal(xt; mu_fc_1*x0 + mu_fc_2*xT, std_fc^2)
        self.mu_fc_1, self.mu_fc_2, var = compute_gaussian_product_coef(
            self.coef_fwd, self.coef_bwd, self.std_fwd, self.std_bwd
        )
        self.std_fc = torch.sqrt(var)

        # forward step conditional: p(xt | x_{t-1}, xT) = Normal(xt; mu_fsc_1*x_{t-1} + mu_fsc_2*xT, std_fsc^2)
        self.mu_fsc_1, self.mu_fsc_2, var = compute_gaussian_product_coef(
            self.coef_onestep, self.coef_bwd, self.std_onestep, self.std_bwd
        )
        self.std_fsc = torch.sqrt(var)

        # signal to noise weight
        self.lambd = (self.mu_fc_1**2) / (
            self.std_fc**2 * snr_threshold + self.mu_fc_1**2
        )

    def get_param_fwd(self, step, xdim=None):
        coef_fwd = self.coef_fwd[step]
        std_fwd = self.std_fwd[step]
        coef_fwd = coef_fwd if xdim is None else unsqueeze_xdim(coef_fwd, xdim)
        std_fwd = std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)
        return coef_fwd, std_fwd

    def get_param_bwd(self, step, xdim=None):
        coef_bwd = self.coef_bwd[step]
        std_bwd = self.std_bwd[step]
        coef_bwd = coef_bwd if xdim is None else unsqueeze_xdim(coef_bwd, xdim)
        std_bwd = std_bwd if xdim is None else unsqueeze_xdim(std_bwd, xdim)
        return coef_bwd, std_bwd

    def get_param_fc(self, step, xdim=None):
        mu_fc_1 = self.mu_fc_1[step]
        mu_fc_2 = self.mu_fc_2[step]
        std_fc = self.std_fc[step]
        mu_fc_1 = mu_fc_1 if xdim is None else unsqueeze_xdim(mu_fc_1, xdim)
        mu_fc_2 = mu_fc_2 if xdim is None else unsqueeze_xdim(mu_fc_2, xdim)
        std_fc = std_fc if xdim is None else unsqueeze_xdim(std_fc, xdim)
        return mu_fc_1, mu_fc_2, std_fc
