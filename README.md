## Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks

This is the implementation of our paper called "Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks" submitted to the [SaTML 2023 Conference](https://satml.org/#). 

---

### Instructions

+ First, install a virtual environment with the required dependencies and create a directory to save the datasets. 

```bash
python3 -m venv env 
source env/bin/activate 
pip install -r requirements.txt
mkdir Datasets # create a directory to save the datasets 
```

+ Experiments can be run in three datasets: CIFAR-10, CIFAR-100 and ImageNet. Datasets are loaded inside the code/eval.py script. By default, RobustBench downloads CIFAR-10 and CIFAR-100 if the datasets do not exist in the prespecified path. However, the ImageNet val set has to be downloaded manually and then post-processed to have the expected directory structure. In order to fix this, follow the instructions from the relevant section at the [RobustBench repository](https://github.com/RobustBench/robustbench). </br>
**Important**: The datasets will be loaded to the Datasets directory. If you already possess the ImageNet val directory, just create a soft link to the Datasets dir:
```bash
ln -s path/to/imagenet_val ./Datasets/val 
```

Then, move to the *code* repository and execute the preparation script, which downloads all the robust models of our study from the RobustBench ModelZoo.
```bash
cd code
python3 prep_models_datasets.py
```

+ The run_eval.py script can be used to evaluate the linf-bounded robustness of any model through PGD. For example, the following line evaluates the Engstrom2019Robustness model from [RobustBench ModelZoo](https://github.com/RobustBench/robustbench). In this case, PGD optimizes the CW loss with a step size of $\alpha = 0.25 * \epsilon$, remaining fixed during the process. The iteration budget is T = 100.

```bash
python3 run_eval.py --model_name Engstrom2019Robustness --loss CW --alpha_eps_ratio 0.25 --step_schedule None --iterations 100 --dataset CIFAR10
```

Results will be printed to the standard output and saved as .json files in the code/results directory.

### Toy Example 

In the *code/toy_example* directory, one can find a notebook reproducing the paper's toy example.