import argparse
import os
from loguru import logger

from eval import eval_pgd
from utils.configs import step_configs, loss_configs
from utils.utils import write_results, print_results
from robustbench import load_model

def parse_arguments():
    """ parser for the script's arguments """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_name", type=str,
                            default='Engstrom2019Robustness',
                            help="The name of robust model to perform robustness "
                            "evaluation through PGD")
    arg_parser.add_argument("--loss", type=str, default='CE',
                            choices=loss_configs.keys(),
                            help="PGD's Surrogate Loss (Default: Cross Entropy)")
    arg_parser.add_argument("--step_schedule", type=str, default='None',
                            choices=step_configs.keys(),
                            help='Step Size Schedule (Default: fixed step size)')
    arg_parser.add_argument("-bs", "--batch_size", type=int, default=50,
                            help='Evaluation batch size (currently hardcoded to 50)')
    arg_parser.add_argument("-a", "--alpha_eps_ratio", type=float, default=0.25,
                            help='step size-perturbation bound ratio')
    arg_parser.add_argument("-T", "--iterations", type=int, default=100,
                            help='num of PGD iterations')
    arg_parser.add_argument("-r", "--restarts", type=int, default=1,
                            help='num of PGD restarts')
    arg_parser.add_argument("-d", "--dataset", type=str, default='CIFAR10',
                            choices=['CIFAR10', 'CIFAR100', 'ImageNet'],
                            help="dataset to run attack")
    arg_parser.add_argument("-o", "--output_folder", type=str, default="results",
                            help="The output folder to write the results of PGD's evaluation")
    arg_parser.add_argument("-mf", "--models_folder", type=str, default="models",
                            help="The folder which contains the robust models"
                                 "(downloaded from RobustBench) to be evaluated")
    arg_parser.add_argument("-df", "--datasets_folder", type=str, default="../Datasets",
                            help="The folder which contains the datasets")

    return vars(arg_parser.parse_args())


if __name__ == '__main__':

    args = parse_arguments()

    model_name = args['model_name']
    epsilon = 0.031 if model_name == 'Zhang2019Theoretically' else 8/255
    dataset = args['dataset']
    epsilon = 4/255 if dataset == "ImageNet" else epsilon

    batch_size, iterations, alpha, restarts = args['batch_size'], args[
        'iterations'], args['alpha_eps_ratio']*epsilon, args['restarts']
    step_schedule = step_configs[args['step_schedule']]
    loss_config = loss_configs[args['loss']]

    # load model from RobustBench
    logger.info(f"Loading model {model_name}")
    classifier = load_model(model_name=model_name, model_dir= args["models_folder"],
                            dataset=dataset.lower(), threat_model='Linf')

    logger.info(f"Running L-inf PGD Attack on {dataset} (T={iterations}, R={restarts}, "\
                f"eps = {epsilon:.4f}, alpha = {alpha:.4f})")
    rob_acc, valids = eval_pgd(classifier, batch_size, iterations,
                               restarts, epsilon, alpha, step_schedule,
                               loss_config, dataset, args['datasets_folder'])

    results_json = os.path.join(
        args["output_folder"], f"{dataset}_{loss_config['name']}.json")

    results_list = [{'T': iterations,
                     'R': restarts, 
                     'alpha': alpha,
                     'step_schedule': args['step_schedule'], 
                     'loss_fn': args['loss']},
                    {'robust_accuracy': 100*rob_acc,
                     'valid_advs_percentage': 100*valids}]

    write_results(model_name, results_json, results_list)
    print_results(dataset, model_name, args['loss'],
                 args['step_schedule'],  rob_acc, valids)
