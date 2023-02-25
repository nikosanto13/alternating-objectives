from typing import Tuple
from robustbench.data import load_cifar100, load_cifar10, load_imagenet
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from utils.utils import validation_check
from PGD import projected_gradient_descent


def eval_PGD(classifier: nn.Module,
             batch_size: int,
             iterations: int,
             restarts: int,
             epsilon: float,
             alpha: float,
             lr_schedule,
             loss_config,
             dataset: str,
             path_to_dataset: str) -> Tuple[float, float]:
    """
    This function is responsible for the PGD evaluation given a specific model.
    It downloads the dataset, and then performs PGD evaluation for the whole dataset.
    It returns the robust accuracy of the model, as well as the percentage of valid adversaries.
    The latter is optional, and it is checked in order to assure that we do not violate any constraint. 
    """

    T, R, eps, alpha, lr_schedule = iterations, restarts, epsilon, alpha, lr_schedule
    loss_fn, kwargs = loss_config['loss_fn'], loss_config['kwargs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 25 if dataset == "ImageNet" else 50

    if dataset == 'CIFAR10':
        x_test, y_test = load_cifar10(
            n_examples=10000, data_dir=path_to_dataset)
    elif dataset == 'CIFAR100':
        x_test, y_test = load_cifar100(
            n_examples=10000, data_dir=path_to_dataset)
    elif dataset == "ImageNet":
        x_test, y_test = load_imagenet(
            n_examples=5000, data_dir=path_to_dataset)

    num_of_batches = int(len(x_test)/batch_size)
    classifier.to(device)

    correct, num_samples = 0, 0
    valids = 0

    for i in tqdm(range(num_of_batches), desc="PGD Evaluation..."):
        x, y = x_test[i*batch_size:(i+1)*batch_size], y_test[i *
                                                             batch_size:(i+1)*batch_size]

        x, y = x.to(device), y.to(device)
        num_samples += x.size(0)
        xcl = x.detach().clone()

        xadv = PGD(classifier, x, y,
                   T=T, restarts=R, alpha=alpha, epsilon=eps,
                   loss_fn=loss_fn, step_schedule=lr_schedule, **kwargs)

        valids += validation_check(xadv, xcl, eps)

        pred = classifier(xadv).argmax(dim=1)
        correct += (y == pred).sum().item()

    # Count robust acc and num of valid examples
    robust_acc = correct/num_samples
    valids_prc = valids/num_samples

    return robust_acc, valids_prc
