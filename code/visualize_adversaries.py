import argparse
from PGD import PGD
from robustbench import load_model
from utils.losses import CE,CW,DLR,Alternate,Linear
from utils.lr_schedules import A3_Schedule,GAMA_Schedule,MD_Schedule,AA_Schedule
from eval_v2 import eval_PGD
import json 


if __name__=='__main__':
    loss = {'loss_fn': Alternate,'kwargs':{'losses':[CE,CW,DLR], 'timesteps': [33,66,100]},'name': 'Alt-CECWDLR'}
    rob_acc, valids = eval_PGD(model_name='Wong2020Fast', batch_size=50, iterations=100, restarts=1, 
            epsilon=4/255, alpha=1/255, lr_schedule=None,loss_config = loss, dataset='ImageNet')
    print(rob_acc,valids)