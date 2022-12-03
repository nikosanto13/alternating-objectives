import argparse
from eval import eval_PGD
import json
from utils.configs import step_configs, loss_configs
import os
from robustbench import load_model

def parse_arguments():
    """ parser for the script's arguments """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_name", type=str, default='Engstrom2019Robustness',
                            help='defense to attack')
    arg_parser.add_argument("--loss", type=str, default='CE', choices=loss_configs.keys(),
                            help="PGD's Surrogate Loss (Default: Cross Entropy)")
    arg_parser.add_argument("--step_schedule", type=str, default='None', choices=step_configs.keys(),
                            help='Step Size Schedule (Default: Fixed Step Size)')
    arg_parser.add_argument("-bs", "--batch_size", type=int, default=50,
                            help='Eval Batch Size (Currently hardcoded to 50)')
    arg_parser.add_argument("-a", "--alpha_eps_ratio", type=float, default=0.25,
                            help='step size-perturbation bound ratio')
    arg_parser.add_argument("-T", "--iterations", type=int, default=100,
                            help='num of iterations')
    arg_parser.add_argument("-r", "--restarts", type=int, default=1,
                            help='num of restarts')
    arg_parser.add_argument("-d", "--dataset", type=str, default='CIFAR10',
                            choices=['CIFAR10', 'CIFAR100', "ImageNet"],
                            help='dataset to run attack')
    arg_parser.add_argument("-o", "--output_folder", type=str, default="results",
                            help="The output folder to write the results of PGD's evaluation")
    arg_parser.add_argument("-mf", "--models_folder", type=str, default="models",
                            help="The folder which contains the robust models (downloaded from RobustBench) to be evaluated")
    arg_parser.add_argument("-df", "--datasets_folder", type=str, default="../Datasets",
                            help="The folder which contains the datasets")
        

    return vars(arg_parser.parse_args())


def write_results(model, path_to_json, list_of_results):
    """ a function to write results of PGD evaluation into jsons """
    
    parent_dir = os.path.dirname(path_to_json)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    try:
        with open(path_to_json, 'r') as fp:
            results = json.load(fp)
            if model not in results.keys():
                results[model] = []
            results[model].append(list_of_results)

    except IOError:
        results = {model: [list_of_results]}

    with open(path_to_json, 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':

    args = parse_arguments()

    model_name = args['model_name']
    epsilon = 0.031 if model_name != 'Zhang2019Theoretically' else 8/255
    dataset = args['dataset']
    epsilon = 4/255 if dataset == "Imagenet" else epsilon

    batch_size, iterations, alpha, restarts = args['batch_size'], args[
        'iterations'], args['alpha_eps_ratio']*epsilon, args['restarts']
    step_schedule = step_configs[args['step_schedule']]
    loss_config = loss_configs[args['loss']]

    # load model from RobustBench
    classifier = load_model(model_name=model_name, model_dir= args["models_folder"],
                            dataset=dataset.lower(), threat_model='Linf')

    rob_acc, valids = eval_PGD(classifier, batch_size, iterations,
                               restarts, epsilon, alpha, step_schedule,
                               loss_config, dataset, args['datasets_folder'])

    results_json = os.path.join(
        args["output_folder"], f"{dataset}_{loss_config['name']}.json")
        
    results_list = [{'T': iterations, 'R': restarts, 'alpha': alpha,
                     'step_schedule': args['step_schedule'], 'loss_fn': args['loss']},
                    {'robust_accuracy': 100*rob_acc, 'valid_advs_percentage': 100*valids}]
    write_results(model_name, results_json, results_list)

    print(f"linf robustness evaluation - {dataset} \n",
          f"\t Model: {model_name}\n",
          f"\t Loss Configuration: {args['loss']}\n",
          f"\t alpha to eps ratio: {args['alpha_eps_ratio']} \n",
          f"\t step size schedule: {args['step_schedule']}  \n \n",
          f"\t Robust Accuracy:  {100*rob_acc:.2f} \n",
          f"\t Rate of valid adversaries:  {100*valids:.2f} \n"
          )
