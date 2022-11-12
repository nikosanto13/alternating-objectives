import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import normalize
from tqdm import tqdm as tqdm 
import math
import random

from PGD import PGD
from robustbench import load_model
from robustbench.data import load_cifar100, load_cifar10, load_imagenet
from utils.losses import CE,CW,DLR,Alternate,Linear
from utils.lr_schedules import A3_Schedule,GAMA_Schedule,MD_Schedule,AA_Schedule


def validation_check(xadv: torch.Tensor, xcl: torch.Tensor, epsilon: float) -> int:
  '''
  this function checks performs a sanity check on whether xadv are valid adversaries of xcl (clean images)
  the check performs three tests: 
    (1) 0 <= xadv <= 1, (2) |xadv - xcl| <= epsilon +- tol, (3) xadv doesn't contain NaN valued pixels 
  '''
  tol = 1e-7
  box = (xadv.flatten(start_dim=1).min(dim=1)[0] >= 0)*(xadv.flatten(start_dim=1).max(dim=1)[0] <= 1)
  eps = ((xadv-xcl).flatten(start_dim=1).min(dim=1)[0] >= -epsilon-tol)*((xadv-xcl).flatten(start_dim=1).max(dim=1)[0] <= epsilon+tol)
  nan = ~xadv.flatten(start_dim=1).isnan().any(dim=1)
  valid = (box*eps*nan).sum().item()
  
  return valid 

def eval_PGD(model_name: str, batch_size: int, iterations: int, restarts: int,
       epsilon: float, alpha: float, lr_schedule, loss_config, dataset: str):

  T, R, eps, alpha, lr_schedule = iterations, restarts, epsilon, alpha, lr_schedule
  loss_fn, kwargs = loss_config['loss_fn'], loss_config['kwargs']
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # prepare cifar10 test data batches
  CIFAR_PATH="../Datasets"
  IMAGENET_PATH="../Datasets"
  batch_size = 50 
  
  if dataset == 'CIFAR10':
    x_test, y_test = load_cifar10(n_examples=10000,data_dir=CIFAR_PATH)
  elif dataset == 'CIFAR100':
    x_test, y_test = load_cifar100(n_examples=10000,data_dir=CIFAR_PATH)   
  elif dataset == "ImageNet":
    x_test, y_test = load_imagenet(n_examples=5000,data_dir=IMAGENET_PATH)
    batch_size = 25 

  num_of_batches = int(len(x_test)/batch_size)

  # load model from RobustBench
  classifier = load_model(model_name=model_name, dataset=dataset.lower(), threat_model='Linf')
  classifier.to(device)

  correct, num_samples = 0,0
  valids = 0 

  for i in tqdm(range(num_of_batches)):
    x,y = x_test[i*batch_size:(i+1)*batch_size], y_test[i*batch_size:(i+1)*batch_size] 
    
    x,y = x.to(device),y.to(device)
    num_samples += x.size(0)
    xcl = x.detach().clone()

    xadv = PGD(classifier,x,y,
           T=T,restarts=R,alpha=alpha,epsilon=eps,
           loss_fn=loss_fn,step_schedule=lr_schedule,**kwargs)

    valids += validation_check(xadv,xcl,eps)

    pred = classifier(xadv).argmax(dim=1)
    correct += (y==pred).sum().item()  

  # Count robust acc and num of valid examples
  robust_acc = correct/num_samples
  valids_prc = valids/num_samples

  return robust_acc, valids_prc 