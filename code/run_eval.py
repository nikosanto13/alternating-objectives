import argparse
from PGD import PGD
from robustbench import load_model
from utils.losses import CE,CW,DLR,Alternate,Linear
from utils.lr_schedules import A3_Schedule,GAMA_Schedule,MD_Schedule,AA_Schedule
from eval import eval_PGD
import json 

# experiments configurations 
# declared here because the loss_fn values are functions from utils.losses
loss_configs = {
  'CE':{'loss_fn':CE,'kwargs':{},'name': 'CE'},
  'CW':{'loss_fn':CW,'kwargs':{},'name': 'CW'},
  'DLR':{'loss_fn':DLR,'kwargs':{},'name': 'DLR'},
  
  'CECW' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,CW], 'timesteps': [50,100]},'name': 'Alt-CECW'},
  'CWCE' :{'loss_fn': Alternate,'kwargs':{'losses':[CW,CE], 'timesteps': [50,100]},'name': 'Alt-CWCE'},

  'CEDLR' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,DLR], 'timesteps': [50,100]},'name': 'Alt-CEDLR'},
  'DLRCE' :{'loss_fn': Alternate,'kwargs':{'losses':[DLR,CE], 'timesteps': [50,100]},'name': 'Alt-DLRCE'},

  'CWDLR' :{'loss_fn': Alternate,'kwargs':{'losses':[CW,DLR], 'timesteps': [50,100]},'name': 'Alt-CWDLR'},
  'DLRCW' :{'loss_fn': Alternate,'kwargs':{'losses':[DLR,CW], 'timesteps': [50,100]},'name': 'Alt-DLRCW'},

  'CECWDLR' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,CW,DLR], 'timesteps': [33,66,100]},'name': 'Alt-CECWDLR'},
  'CEDLRCW' :{'loss_fn': Alternate,'kwargs':{'losses':[CE,DLR,CW], 'timesteps': [33,66,100]},'name': 'Alt-CEDLRCW'},

  'Conv1_CECW' :{'loss_fn': Linear,'kwargs':{'losses':[CE,CW], 'coeffs': [0.25,0.75]}, 'name': 'Conv-CECW(0.25/0.75)'},
  'Conv2_CECW' :{'loss_fn': Linear,'kwargs':{'losses':[CE,CW], 'coeffs': [0.75,0.25]}, 'name': 'Conv-CECW(0.75/0.25)'}  
}

# dictionary that maps arguments to step size schedule functions 
step_configs = {
  'None': None,
  'A3': A3_Schedule,
  'GAMA-PGD': GAMA_Schedule,
  'MD-Attack': MD_Schedule,
  'AutoAttack': AA_Schedule
}

if __name__=='__main__':

  # read arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_name" , type=str, default='Engstrom2019Robustness', help='defense to attack')
  ap.add_argument("--loss", type=str, default='CE', choices=loss_configs.keys(), help="PGD's Surrogate Loss (Default: Cross Entropy)")
  ap.add_argument("--step_schedule", type=str, default='None', choices=step_configs.keys(), help='Step Size Schedule (Default: Fixed Step Size)')
  ap.add_argument("--batch_size", type=int, default=50, help='Eval Batch Size (Currently hardcoded to 50)')
  ap.add_argument("--alpha_eps_ratio", type=float, default=0.25, help='step size-perturbation bound ratio')
  ap.add_argument("--iterations", type=int, default=100, help='num of iterations')
  ap.add_argument("--restarts", type=int, default=1, help='num of restarts')
  ap.add_argument("--dataset", type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100',"ImageNet"],help='dataset to run attack')


  args = vars(ap.parse_args())
  model_name  = args['model_name']
  epsilon = 8/255 if model_name != 'Zhang2019Theoretically' else 0.031
  dataset = args['dataset']
  if dataset=='ImageNet':
    epsilon = 4/255

  batch_size, iterations, alpha, restarts = args['batch_size'], args['iterations'], args['alpha_eps_ratio']*epsilon , args['restarts']
  step_schedule = step_configs[args['step_schedule']]
  loss_config = loss_configs[args['loss']]
  

  rob_acc, valids = eval_PGD(model_name, batch_size, iterations, restarts, epsilon, alpha, step_schedule, loss_config, dataset)
  results_dict_name = f"results/{dataset}_{loss_config['name']}.json"

  results_list = [{'T': iterations, 'R': restarts, 'alpha': alpha, 'step_schedule': args['step_schedule'], 'loss_fn': args['loss']},
          {'robust_accuracy': 100*rob_acc, 'valid_advs_percentage': 100*valids}]
  
  print(f"linf robustness evaluation - {dataset} \n", 
        f"\t Model: {model_name}\n",
        f"\t Loss Configuration: {args['loss']}\n",
        f"\t alpha to eps ratio: {args['alpha_eps_ratio']} \n",
        f"\t step size schedule: {args['step_schedule']}  \n \n",
        f"\t Robust Accuracy:  {100*rob_acc:.2f} \n",
        f"\t Rate of valid adversaries:  {100*valids:.2f} \n"
        ) 

  try:
      with open(results_dict_name, 'r') as fp:
          results = json.load(fp)
          if model_name in results.keys():
            results[model_name].append(results_list)
          else:
            results[model_name] = [results_list] 

  except IOError:
      results = {model_name: [results_list]}

  with open(results_dict_name, 'w') as fp:
      json.dump(results, fp)
  