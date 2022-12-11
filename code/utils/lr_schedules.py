import math
import numpy as np


def a3_schedule(alpha, **kwargs):
    """ From paper : Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack """
    t, total_its, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
    return 1/2*epsilon*(1+math.cos((t % total_its)/total_its*math.pi))


def gama_schedule(alpha, **kwargs):
    """ Guided Adversarial step schedule  : https://arxiv.org/pdf/2011.14969.pdf """
    t, total_its, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
    t1, t2 = int(0.6*total_its), int(0.85*total_its)  # for T=100, (t1,t2) = (60,85)

    if t < t1:
        return alpha

    if t >= t1 and t < t2:
        return alpha/10
    else:
        return alpha/100


def md_schedule(alpha, **kwargs):
    """
    Margin Decomposition Schedule, https://arxiv.org/pdf/2006.13726.pdf
    The whole duration is divided into N intervals of duration T/N
    In each interval, we desire the step size to be cosine-annealed from 2*epsilon to 0
    """

    t, total_its, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
    num_stages = kwargs['N']

    dur = total_its/num_stages
    offset = t // dur  # E.g. if T=100, N=3 > For t=20: offset = 0, t=40: offset = 1
    freq = (t-offset*dur)/dur
    return epsilon*(1+math.cos(freq*math.pi))


def aa_schedule(alpha, **kwargs):
    """ AutoAttack schedule, https://arxiv.org/pdf/2003.01690.pdf
    However, in AA, the step size is changed based on the optimization progress.
    Here, we do not monitor the progress, and the step size is reduced regardless.
    """

    t, total_its, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
    p = [0, 0.22]
    while p[-1] < 1.:
        x = p[-1] + max(p[-1]-p[-2]-0.03, 0.06)
        p.append(x)
    p = p[:-1]
    p[-1] = 1.
    checkpoints = (total_its*np.array(p)).astype(int)
    # check in which checkpoint interval does k belongs
    k = np.where(t < checkpoints)[0][0] - 1

    return 2*epsilon/2**k
