import torch

def validation_check(xadv: torch.Tensor, xcl: torch.Tensor, epsilon: float) -> int:
    '''
    this function checks performs a sanity check on whether xadv are valid adversaries of xcl (clean images)
    the check performs three tests: 
      (1) 0 <= xadv <= 1, (2) |xadv - xcl| <= epsilon +- tol, (3) xadv doesn't contain NaN valued pixels 
    '''
    tol = 1e-7

    box = (xadv.flatten(start_dim=1).min(dim=1)[
           0] >= 0)*(xadv.flatten(start_dim=1).max(dim=1)[0] <= 1)
    eps = ((xadv-xcl).flatten(start_dim=1).min(dim=1)[0] >= -epsilon-tol)*(
        (xadv-xcl).flatten(start_dim=1).max(dim=1)[0] <= epsilon+tol)
    nan = ~xadv.flatten(start_dim=1).isnan().any(dim=1)
    valid = (box*eps*nan).sum().item()

    return valid
