import torch

from runners.conditional_bridge import q_sample as cb_q_sample
from runners.ddpm import p_posterior as ddpm_p_posterior

def take_dim(x, start, length):
    stop = start + length
    return x[:, start:stop, :, :]


def q_sample(diffusion, step, x0, xT):
    """sample x_t^1:K ~ q(x_t^1:K | x_0, x_T^1:K)
    xt^k = mu_fc_1*x0 + mu_fc_2*sum_k xT^k + epsilon^k*std_fc"""

    K = diffusion.n_sources
    img_channels = xT.shape[1] // K
    sources_channels = xT.shape[1]

    assert x0[:, 0, :, :].shape == xT[:, 0, :, :].shape
    assert sources_channels == K * img_channels

    xt = []
    for k_channels_start in range(0, sources_channels, img_channels):
        xt.append(
            cb_q_sample(
                diffusion=diffusion,
                step=step,
                x0=x0,
                xT=take_dim(xT, k_channels_start, img_channels), # not used if ddpm
            )
        )

    return torch.cat(xt, dim=1)



def compute_label(diffusion, step, x0, xt, compute_label_fn, xT = None):
    """
    Compute the label for each source channel.
    - if q_sample_fn comes from conditional_bridge (conditional training):
        label = (xt^k - mu_fc_1*x0 -mu_fc_2*sum_k xT^k ) / std_fc
    - if q_sample_fn comes from ddpm (marginal training):
        label = (xt^k - coef_fwd * x0 ) / std_fwd
    """

    K = diffusion.n_sources
    img_channels = xT.shape[1] // K
    sources_channels = xT.shape[1]

    assert x0[:, 0, :, :].shape == xT[:, 0, :, :].shape
    assert sources_channels == K * img_channels

    label = []
    for k_channels_start in range(0, sources_channels, img_channels):
        res = compute_label_fn(
            diffusion=diffusion,
            step=step,
            x0=x0,
            xt=take_dim(xt, k_channels_start, img_channels),
            xT = take_dim(xT, k_channels_start, img_channels) if xT is not None else None # not used if ddpm
        )
        label.append(res)

    return torch.cat(label, dim=1)


def compute_pred_x0(diffusion, step, xt, net_out, compute_pred_x0_fn, xT = None):
    """
    Compute the predicted x0.
        - if q_sample_fn comes from conditional_bridge (conditional estimator):
            pred_x0 = hat{x}0^cond = E[x0| xt^1:K, xT^1:K] 
            = 1/ K * sum_k ( xt^k - mu_fc_2 * xT^k - std_fc * net_out[k] ) / mu_fc_1 
        - if q_sample_fn comes from ddpm (marginal estimator):
            pred_x0 = hat{x}0^marg = E[x0| xt^1:K] 
            = 1/ K * sum_k ( xt^k - std_fc * net_out[k] ) / std_fwd 
    """

    K = diffusion.n_sources
    img_channels = xt.shape[1] // K
    sources_channels = xt.shape[1]

    assert sources_channels == K * img_channels

    # we loop over the sources channels and concatenate
    pred_x0_sum = None
    for k_channels_start in range(0, sources_channels, img_channels):

        current = compute_pred_x0_fn(
            diffusion=diffusion,
            step=step,
            xt=take_dim(xt, k_channels_start, img_channels),
            net_out=take_dim(net_out, k_channels_start, img_channels),
            xT = take_dim(xT, k_channels_start, img_channels) if xT is not None else None # not used if ddpm
        )
        if pred_x0_sum is None:
            pred_x0_sum = current
        else:
            pred_x0_sum = pred_x0_sum + current

    return pred_x0_sum / K


def p_posterior(diffusion, prev_step, step, xt, x0):
    """x_{t-1}^k ~ p(x_{t-1}^k | x_t^k x_0), for k = 1:K, 0 < t <= T"""

    assert prev_step < step

    K = diffusion.n_sources
    img_channels = xt.shape[1] // K
    sources_channels = xt.shape[1]

    assert sources_channels == K * img_channels

    outs = []

    # we loop over the sources channels and concatenate
    for k_channels_start in range(0, sources_channels, img_channels):
        out = ddpm_p_posterior(
            diffusion=diffusion,
            prev_step=prev_step,
            step=step,
            xt=take_dim(xt, k_channels_start, img_channels),
            x0=x0,
        )

        outs.append(out)

    return torch.cat(outs, dim=1)

