import torch
import torch.nn as nn


def CE(x, y, **kwargs):
    # kwargs are not needed here, just for implementation convenience
    return nn.CrossEntropyLoss(reduction='none')(x, y)


def CW(x, y, **kwargs):
    # kwargs are not needed here, just for implementation convenience
    px = x
    px_sorted, ind_sorted = px.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    ui = torch.arange(px.shape[0])

    return -(px[ui, y] - px_sorted[:, -2] * ind - px_sorted[:, -1] * (1. - ind))


def MSE(x, y, **kwargs):
    # kwargs are not needed here, just for implementation convenience
    z0 = kwargs['z0']
    soft = nn.Softmax(dim=1)
    p0 = soft(z0)
    p = soft(x)
    return torch.norm(p0-p, p=2, dim=1)


def DLR(x, y, **kwargs):
    # kwargs are not needed here, just for implementation convenience
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] *
             (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def Alternate(x, y, **kwargs):
    # Alternate losses during PGD

    it = kwargs['it']
    timesteps = kwargs['timesteps']
    losses = kwargs['losses']

    assert len(timesteps) == len(
        losses), 'Num of loss functions should be the same with num of timesteps where switching occurs (+ the timestep t=T)'

    for i, T in enumerate(timesteps):
        if it < T:
            return losses[i](x, y, **kwargs)


def Linear(x, y, **kwargs):
    '''
    Linear combination loss
    args (that need to be specified by user):
      losses (list): list of functions that return Bx1 tensors of loss values
      coeffs (list): list of weighting coefficients for every loss in losses
      decay_timesteps (list): decaying factor of the coefficients (optional)

    E.g. for convex combination: Linear(x,y,losses=[CE,CW],coeffs=[0.25,0.75])
       for GAMA-PGD:           Linear(x,y,losses=[CW,MSE],coeffs=[1,1],decay=[0,20])
    '''
    assert len(
        kwargs['losses']) == 2, 'Currently, this loss function is only supported for two objectives'
    assert len(kwargs['losses']) == len(
        kwargs['coeffs']), 'Every loss functions should have a corresponding coefficient'

    loss0, loss1 = kwargs['losses']
    coeff0, coeff1 = kwargs['coeffs']

    t = kwargs['it']

    if 'decay' in kwargs.keys():
        assert len(
            kwargs['decay']) == 2, 'Num of decaying timesteps is not equal to num of losses'
        tau0, tau1 = kwargs['decay']
        c0 = coeff0 if tau0 == 0 else max(coeff0-t*coeff0/tau0, 0)
        c1 = coeff1 if tau1 == 0 else max(coeff1-t*coeff1/tau1, 0)
    else:
        c0, c1 = coeff0, coeff1
    return c0*loss0(x, y, **kwargs) + c1*loss1(x, y, **kwargs)
