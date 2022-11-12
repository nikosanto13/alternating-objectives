cd ../
id=0

for MODEL in Engstrom2019Robustness Carmon2019Unlabeled Hendrycks2019Using Zhang2019You Zhang2019Theoretically\
   Wu2020Adversarial Sehwag2021Proxy_R18 Andriushchenko2020Understanding Dai2021Parameterizing\
   Gowal2021Improving_28_10_ddpm_100m Huang2021Exploring_ema Zhang2020Geometry\
   Rade2021Helper_R18_extra Addepalli2021Towards_RN18 Sehwag2020Hydra 
do
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CE --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss DLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CEDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR10
    echo ${MODEL} Done
done 

