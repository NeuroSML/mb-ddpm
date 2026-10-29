import torch


def q_sample(diffusion, step, x0, xT):
    """sample x_t ~ q(x_t | x_0, x_T)
    where xt = mu_fc_1*x0 + mu_fc_2*xT + epsilon*std_fc"""

    assert x0.shape == xT.shape

    mu_fc_1, mu_fc_2, std_fc = diffusion.get_param_fc(step, xdim=xT.shape[1:])

    xt = mu_fc_1 * x0 + mu_fc_2 * xT
    xt = xt + std_fc * torch.randn_like(xt)
    
    return xt.detach()


def compute_label(diffusion, step, x0, xt, xT):
    """compute label = (xt - mu_fc_1*x0 -mu_fc_2*xT ) / std_fc"""

    assert x0.shape == xT.shape
    
    mu_fc_1, mu_fc_2, std_fc = diffusion.get_param_fc(step, xdim=xt.shape[1:])

    label = (xt - mu_fc_1 * x0 - mu_fc_2 * xT) / std_fc

    return label.detach()



def compute_pred_x0(diffusion, step, xt, xT, net_out):
    """compute E[x0| xt, xT] = (xt - net_out * std_fc - mu_fc_2 * xT) / mu_fc_1"""

    mu_fc_1, mu_fc_2, std_fc = diffusion.get_param_fc(step, xdim=xt.shape[1:])

    return (xt - net_out * std_fc - mu_fc_2 * xT) / mu_fc_1
