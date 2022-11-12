import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import normalize

from PGD import PGD_distances
from robustbench import load_model
from robustbench.data import load_cifar100, load_cifar10, load_imagenet
from utils.losses import CE,CW,DLR,Alternate,Linear
from utils.lr_schedules import A3_Schedule,GAMA_Schedule,MD_Schedule,AA_Schedule

from tqdm import tqdm as tqdm 
import math
import random
import numpy as np
import matplotlib.pyplot as plt 

def plot_cos_similarity(model_name: str,
       batch_size: int,
       iterations: int,
       restarts: int,
       epsilon: float, 
       alpha: float,
       lr_schedule,
       loss_config,
       dataset):

  T, R, eps, alpha, lr_schedule = iterations, restarts, epsilon, alpha, lr_schedule
  loss_fn, kwargs = loss_config['loss_fn'], loss_config['kwargs']

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # prepare cifar10 test data batches
  batch_size = 100
  CIFAR_PATH='./data'

  x_test, y_test = load_cifar10(n_examples=100,data_dir=CIFAR_PATH)
  num_of_batches = int(len(x_test)/batch_size)

  classifier = load_model(model_name=model_name, dataset=dataset.lower(), threat_model='Linf')
  classifier.to(device)

  correct, num_samples = 0,0
  valids = 0 

  for i in tqdm(range(num_of_batches)):
    x,y = x_test[i*batch_size:(i+1)*batch_size], y_test[i*batch_size:(i+1)*batch_size] 
    
    x,y = x.to(device),y.to(device)
    num_samples += x.size(0)
    xcl = x.detach().clone()

    xadv,dists = PGD_distances(classifier,x,y,
           T=T,restarts=R,alpha=alpha,epsilon=eps,
           loss_fn=loss_fn,step_schedule=lr_schedule,**kwargs)
  
  return dists

if __name__=='__main__': 
    model_name = 'Hendrycks2019Using'
    loss_configs = {
        'CE':{'loss_fn':CE,'kwargs':{},'name': 'CE'},
        'CW':{'loss_fn':CW,'kwargs':{},'name': 'CW'},
        'DLR':{'loss_fn':DLR,'kwargs':{},'name': 'DLR'},
        'CECW' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,CW], 'timesteps': [50,100]},'name': 'Alt-CECW'},
        'CEDLR' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,DLR], 'timesteps': [50,100]},'name': 'Alt-CEDLR'},
        'CECWDLR' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,CW,DLR], 'timesteps': [33,66,100]},'name': 'Alt-CECWDLR'},
    }

    for loss in loss_configs.keys():
        dists = plot_cos_similarity(model_name,50,100,1,8/255,2/255,None,loss_configs[loss],'CIFAR10')
        dists = torch.flatten(dists,2,4)
        dist1, dist2 = dists[1:], dists[:-1]
        cos = nn.CosineSimilarity(dim=2,eps=1e-6)
        avg_sim = cos(dist1,dist2).mean(dim=1)
        t = np.arange(2,101)
        plt.plot(t,avg_sim.cpu().tolist(),label=loss)
    plt.legend()
    plt.savefig(f'images/{model_name}')