import tqdm
from robustbench import load_model
import os

if __name__ == "__main__":

    # CIFAR10 models
    models = [
        'Engstrom2019Robustness',
        'Carmon2019Unlabeled',
        'Hendrycks2019Using',
        'Zhang2019You',
        'Zhang2019Theoretically',
        'Wu2020Adversarial',
        'Sehwag2021Proxy_R18', 
        'Andriushchenko2020Understanding',
        'Dai2021Parameterizing',
        'Gowal2021Improving_28_10_ddpm_100m',
        'Huang2021Exploring_ema', 
        'Zhang2020Geometry', 
        'Rade2021Helper_R18_extra',
        'Addepalli2021Towards_RN18',
        'Sehwag2020Hydra'
    ]
    
    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="cifar10", threat_model="Linf")
        except:
            print(f"skip: {model}")

    models = [
        'Rade2021Helper_R18_ddpm',
        'Rebuffi2021Fixing_R18_ddpm',
        'Addepalli2021Towards_PARN18',
        'Rice2020Overfitting',
        'Hendrycks2019Using',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm'
    ]

    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="cifar100", threat_model="Linf")
        except:
            print(f"skip: {model}")

    models = [
        "Salman2020Do_R18",
        "Salman2020Do_R50",
        "Engstrom2019Robustness",
        "Wong2020Fast",
    ]

    for model in tqdm.tqdm(models):
        try:
            load_model(model, dataset="imagenet", threat_model="Linf")
        except:
            print(f"skip: {model}")