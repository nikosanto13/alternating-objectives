# a script to download the models to the specified folder

import tqdm
from robustbench import load_model
import sys

if __name__ == "__main__":
    path_to_models = sys.argv[1]
    if not path_to_models:
        path_to_models = "./models"

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
            load_model(model, model_dir= path_to_models, dataset="cifar10", threat_model="Linf")
        except:
            print(f"skip: {model}")

    models = [
        'Rade2021Helper_R18_ddpm',
        'Rebuffi2021Fixing_R18_ddpm',
        'Addepalli2021Towards_PARN18',
        'Rice2020Overfitting',
        'Hendrycks2019Using',
        'Rebuffi2021Fixing_28_10_cutmix_ddpm',
        'Addepalli2022Efficient_WRN_34_10',
        'Sehwag2021Proxy',
        'Pang2022Robustness_WRN28_10',
        'Jia2022LAS-AT_34_10'
    ]

    for model in tqdm.tqdm(models):
        try:
            load_model(model, model_dir= path_to_models, dataset="cifar100", threat_model="Linf")
        except:
            print(f"skip: {model}")

    models = [
        'Salman2020Do_R18',
        'Salman2020Do_R50',
        'Engstrom2019Robustness',
        'Wong2020Fast',
        'Salman2020Do_50_2',
        'Debenedetti2022Light_XCiT-S12'
    ]

    for model in tqdm.tqdm(models):
        try:
            load_model(model, model_dir= path_to_models, dataset="imagenet", threat_model="Linf")
        except:
            print(f"skip: {model}")
