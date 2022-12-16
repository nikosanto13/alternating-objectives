cd ../
id=0

for MODEL in Rade2021Helper_R18_ddpm Rebuffi2021Fixing_R18_ddpm Addepalli2021Towards_PARN18 Rice2020Overfitting\
             Hendrycks2019Using Rebuffi2021Fixing_28_10_cutmix_ddpm Addepalli2022Efficient_WRN_34_10\
             Sehwag2021Proxy Pang2022Robustness_WRN28_10 Jia2022LAS-AT_34_10
do
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CE --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss DLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CEDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    echo ${MODEL} Done
done 

