import torch
import torch.nn as nn
import random

def ce_loss(img, label, **kwargs):
    """ Interface for cross-entropy (CE) loss
    """
    return nn.CrossEntropyLoss(reduction='none')(img, label)


def cw_loss(img, label, **kwargs):
    """ Typical Carlini-Wagner (CW) loss
    """
    logits = img
    logits_sorted, ind_sorted = logits.sort(dim=1)
    ind = (ind_sorted[:, -1] == label).float()
    idx = torch.arange(px.shape[0])

    return -(logits[idx, label] - logits_sorted[:, -2] * ind - logits_sorted[:, -1] * (1. - ind))


def MSE(img, label, **kwargs):
    """ Mean Squared Error (MSE) between the logits.
    """
    logits_clean = kwargs['z0']
    soft = nn.Softmax(dim=1)
    probs_clean = soft(logits_clean)
    probs = soft(img)

    return torch.norm(probs_clean-probs, p=2, dim=1)

def dlr_loss(img, label, **kwargs):
    """ Difference of Logits Ratio (DLR) loss
    Implementation from Croce
    """
    x_sorted, ind_sorted = img.sort(dim=1)
    ind = (ind_sorted[:, -1] == label).float()
    idx = torch.arange(img.shape[0])

    return -(img[idx, label] - x_sorted[:, -2] * ind - x_sorted[:, -1] *
             (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

def alternate_loss(img, label, **kwargs):
    """ Alternates between different losses
    """

    it = kwargs['it']
    timesteps = kwargs['timesteps']
    losses = kwargs['losses']
    assert len(timesteps) == len(
        losses), 'Num of loss functions should be the same with num of timesteps"\
                 " where switching occurs (+ the timestep t=T)'

    for i, num_iter in enumerate(timesteps):
        if it < num_iter:
            return losses[i](img, label, **kwargs)

def random_alternate_loss(img, label, **kwargs):
    """ Randomly alternates between CE and DLR losses
    """

    prob = random.random()

    if prob > 0.5:
        return ce_loss(img,label,**kwargs)
    else:
        return dlr_loss(img,label,**kwargs)

def linear_loss(img, label, **kwargs):
    """
    Linear combination loss
    args (that need to be specified by user):
      losses (list): list of functions that return Bx1 tensors of loss values
      coeffs (list): list of weighting coefficients for every loss in losses
      decay_timesteps (list): decaying factor of the coefficients (optional)

    E.g. for convex combination: Linear(img,label,losses=[CE,CW],coeffs=[0.25,0.75])
       for GAMA-PGD:           Linear(img,label,losses=[CW,MSE],coeffs=[1,1],decay=[0,20])
    """

    assert len(
        kwargs['losses']) == 2, 'Currently, this loss function is only supported for two objectives'
    assert len(kwargs['losses']) == len(
        kwargs['coeffs']), 'Every loss functions should have a corresponding coefficient'

    loss0, loss1 = kwargs['losses']
    coeff0, coeff1 = kwargs['coeffs']

    num_iter = kwargs['it']

    if 'decay' in kwargs:
        assert len(
            kwargs['decay']) == 2, 'Num of decaying timesteps is not equal to num of losses'
        tau0, tau1 = kwargs['decay']
        alpha_0 = coeff0 if tau0 == 0 else max(coeff0-num_iter*coeff0/tau0, 0)
        alpha_1 = coeff1 if tau1 == 0 else max(coeff1-num_iter*coeff1/tau1, 0)
    else:
        alpha_0, alpha_1 = coeff0, coeff1
 
    return alpha_0*loss0(img, label, **kwargs) + alpha_1*loss1(img, label, **kwargs)
