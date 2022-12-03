import torch
from utils.losses import CE, CW, DLR, Alternate, Linear
from torch.autograd import Variable
INF = float('INF')

def PGD(model, x, y, T, restarts, alpha, epsilon, loss_fn,
        init='id', step_schedule=None, return_last=False, **kwargs):
    # Only l-inf. Implemented according to https://github.com/ermongroup/ODS/blob/master/whitebox_pgd_attack_cifar10_ODI.py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    x_adv = x.clone()

    # N is the number of stages for each PGD variant. E.g. when we use a single loss during PGD, then N = 1
    N = len(kwargs['losses']) if (
        loss_fn == Alternate or loss_fn == Linear) else 1
    kwargs['T'] = T

    for _ in range(restarts):
        xn = Variable(x, requires_grad=True)

        if init == 'random' or restarts > 1:
            random_noise = torch.FloatTensor(
                xn.shape).uniform_(-epsilon, epsilon).to(device)
            random_noise = epsilon*torch.sign(random_noise)
            xn = Variable(xn.data + random_noise, requires_grad=True)

        alpha_t = alpha
        for t in range(T):
            out = model(xn)
            if t == 0:
                z0 = out.clone().detach()
                kwargs['z0'] = z0

            # Add iteration count on kwargs
            kwargs['it'] = t
            loss_fn(out, y, **kwargs).mean().backward()

            # check the 0-1 loss and update the x_adv in case of misclassification
            is_fooled = (out.argmax(dim=1) != y)
            x_adv[is_fooled] = xn[is_fooled]

            # Schedule the step size
            kwargs2 = {'it': t, 'T': T, 'epsilon': epsilon, 'N': N}
            alpha_t = step_schedule(
                alpha, **kwargs2) if step_schedule else alpha

            # follow the gradient direction
            eta = alpha_t * xn.grad.data.sign()
            xn = Variable(xn.data + eta, requires_grad=True)

            # project the perturbation
            eta = torch.clamp(xn.data - x.data, -epsilon, epsilon)

            xn = Variable(x.data + eta, requires_grad=True)
            xn = Variable(torch.clamp(xn, 0, 1.0), requires_grad=True)

        # check the 0-1 loss once again, in case where the last iteration generates a successful perturbation
        out = model(xn)
        is_fooled = (out.argmax(dim=1) != y)
        x_adv[is_fooled] = xn[is_fooled]

    xr = xn.data if return_last else x_adv.data
    return xr
