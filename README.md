## Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks

This is the implementation of our paper called "Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks" submitted to the [SaTML 2023 Conference](https://satml.org/#). 

---

### Instructions

+ First, install the required libraries. Then, move to the *code* repository and execute the preparation script, which downloads the CIFAR10 dataset and the robust models of our study.

```bash
python3 -m venv env 
source venv/bin/activate 
pip install -r requirements.txt 
cd code
python3 prep_models_datasets.py
```

+ The run_eval.py script can be used to evaluate the linf-bounded robustness of any model through PGD. For example, the following line evaluates the Engstrom2019Robustness model from [RobustBench ModelZoo](https://github.com/RobustBench/robustbench). In this case, PGD optimizes the CW loss with a step size of $\alpha = 0.25 * \epsilon$, remaining fixed during the process. The iteration budget is T = 100. 


```bash
python3 run_eval.py --model_name Engstrom2019Robustness --loss CW --alpha_eps_ratio 0.25 --step_schedule None --iterations 100
```

Results will be printed to the standard output and saved as .json files in the code/results/ directory.

### Toy Example 

In the *code/toy_example* directory, one can find a notebook reproducin the paper's toy example.