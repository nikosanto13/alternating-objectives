import math
import numpy as np
import sys

def A3_Schedule(alpha,**kwargs):
  # From paper : Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack
  t, T, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
  return 1/2*epsilon*(1+math.cos((t % T)/T*math.pi))

def GAMA_Schedule(alpha,**kwargs):
  # Guided Adversarial  : https://arxiv.org/pdf/2011.14969.pdf
  t, T, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
  t1, t2 = int(0.6*T), int(0.85*T) #for T=100, (t1,t2) = (60,85)
  if t < t1:
    return alpha
  elif t >= t1 and t < t2:
    return alpha/10
  else:
    return alpha/100

def MD_Schedule(alpha,**kwargs):
  # Margin Decomposition Schedule, https://arxiv.org/pdf/2006.13726.pdf
  # The whole duration is divided into N intervals of duration T/N
  # In each interval, we desire the step size to be cosine-annealed from 2*epsilon to 0 
  t, T, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
  N = kwargs['N'] 

  dur = T/N
  offset =  t // dur # E.g. if T=100, N=3 > For t=20: offset = 0, t=40: offset = 1 
  freq = (t-offset*dur)/dur
  return epsilon*(1+math.cos(freq*math.pi))

def AA_Schedule(alpha,**kwargs):
  # AutoAttack Schedule, https://arxiv.org/pdf/2003.01690.pdf
  # Beware of this though: In AutoAttack, the step size is changed based on the optimization progress, not always
  t, T, epsilon = kwargs['it'], kwargs['T'], kwargs['epsilon']
  p = [0,0.22]
  while p[-1] < 1.:
    x = p[-1] + max(p[-1]-p[-2]-0.03,0.06)
    p.append(x)
  p = p[:-1]
  p[-1] = 1.
  checkpoints = (T*np.array(p)).astype(int)
  k = np.where(t < checkpoints)[0][0] - 1 # check in which checkpoint interval does k belongs

  return 2*epsilon/2**k