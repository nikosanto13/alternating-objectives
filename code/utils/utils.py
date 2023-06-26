import os
import json
import torch
from loguru import logger

def validation_check(xadv: torch.Tensor, xcl: torch.Tensor, epsilon: float) -> int:
    """ this function checks performs a sanity check on whether xadv are 
    valid adversaries of xcl (clean images). The check performs three tests: 
      (1) 0 <= xadv <= 1, 
      (2) |xadv - xcl| <= epsilon +- tol, 
      (3) xadv doesn't contain NaN valued pixels 
    """
    tol = 1e-7

    box = (xadv.flatten(start_dim=1).min(dim=1)[
           0] >= 0)*(xadv.flatten(start_dim=1).max(dim=1)[0] <= 1)
    eps = ((xadv-xcl).flatten(start_dim=1).min(dim=1)[0] >= -epsilon-tol)*(
        (xadv-xcl).flatten(start_dim=1).max(dim=1)[0] <= epsilon+tol)
    nan = ~xadv.flatten(start_dim=1).isnan().any(dim=1)
    valid = (box*eps*nan).sum().item()

    return valid

def write_results(model, path_to_json, list_of_results):
    """ a function to write results of PGD evaluation into jsons """

    parent_dir = os.path.dirname(path_to_json)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    try:
        with open(path_to_json, 'r') as fp:
            results = json.load(fp)
            results.setdefault(model,[]).append(list_of_results)

    except IOError:
        results = {model: [list_of_results]}

    with open(path_to_json, 'w') as fp:
      json.dump(results, fp)

def print_results(dataset, model_name, loss,
                  step_schedule, robust_accuracy, valids):
    """ logging results """

    logger.info(f"Linf robustness evaluation for {model_name} - {dataset}")
    logger.info(f"\t loss configuration: {loss}")
    logger.info(f"\t step size schedule: {step_schedule}")
    logger.info(f"\t rob_accuracy = {100*robust_accuracy:.2f}%")
    logger.info(f"\t valid_adversaries = {100*valids:.2f}%")
