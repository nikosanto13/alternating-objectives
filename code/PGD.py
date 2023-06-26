import torch
import torch.nn as nn
from utils.losses import alternate_loss, linear_loss
from torch.autograd import Variable
INF = float('INF')


def projected_gradient_descent(model: nn.Module,
                               img: torch.Tensor,
                               label: torch.Tensor,
                               num_iter: int,
                               restarts: int,
                               alpha: float,
                               epsilon: float,
                               loss_fn,
                               init: str = 'id',
                               step_schedule=None,
                               return_last: bool = False,
                               **kwargs) -> torch.Tensor: #pylint: disable=too-many-arguments
    """Only l-inf. Implemented according to 
        https://github.com/ermongroup/ODS/blob/master/whitebox_pgd_attack_cifar10_ODI.py
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    x_adv = img.clone()

    # N is the number of stages for each PGD variant.
    # E.g. when we use a single loss during PGD, then N = 1
    n_stages = len(kwargs['losses']) if loss_fn in (
        alternate_loss, linear_loss) else 1
    kwargs['T'] = num_iter

    for _ in range(restarts):
        x_n = Variable(img, requires_grad=True)

        if init == 'random' or restarts > 1:
            random_noise = torch.FloatTensor(
                x_n.shape).uniform_(-epsilon, epsilon).to(device)
            random_noise = epsilon*torch.sign(random_noise)
            x_n = Variable(x_n.data + random_noise, requires_grad=True)

        alpha_t = alpha
        for it in range(num_iter):
            out = model(x_n)
            if it == 0:
                logits_clean = out.clone().detach()
                kwargs['z0'] = logits_clean

            # Add iteration count on kwargs
            kwargs['it'] = it
            loss_fn(out, label, **kwargs).mean().backward()

            # check the 0-1 loss and update the x_adv in case of misclassification
            is_fooled = out.argmax(dim=1) != label
            x_adv[is_fooled] = x_n[is_fooled]

            # Schedule the step size
            kwargs2 = {'it': it, 'T': num_iter, 'epsilon': epsilon, 'N': n_stages}
            alpha_t = step_schedule(
                alpha, **kwargs2) if step_schedule else alpha

            # follow the gradient direction
            eta = alpha_t * x_n.grad.data.sign()
            x_n = Variable(x_n.data + eta, requires_grad=True)

            # project the perturbation
            eta = torch.clamp(x_n.data - img.data, -epsilon, epsilon)

            x_n = Variable(img.data + eta, requires_grad=True)
            x_n = Variable(torch.clamp(x_n, 0, 1.0), requires_grad=True)

        # check the 0-1 loss once again, in case where
        # the last iteration generates a successful perturbation
        out = model(x_n)
        is_fooled = out.argmax(dim=1) != label
        x_adv[is_fooled] = x_n[is_fooled]

    x_adv_final = x_n.data if return_last else x_adv.data
 
    return x_adv_final
