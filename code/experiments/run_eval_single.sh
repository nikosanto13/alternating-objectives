cd ../

for MODEL in Rebuffi2021Fixing_28_10_cutmix_ddpm
do
    echo Waiting for free GPU ... ${MODEL}
    id=0
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)

    while [ $free_mem -lt 6000 ]; do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
        sleep 2
    done
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CE --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss DLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CEDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset CIFAR100
    echo ${MODEL} Done
done 
