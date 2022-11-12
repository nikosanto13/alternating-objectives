cd ../
id=0

for MODEL in Salman2020Do_R18 Salman2020Do_R50 Engstrom2019Robustness Wong2020Fast
do
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CE --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss DLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECW --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CEDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    CUDA_VISIBLE_DEVICES=$id python3 run_eval.py --model_name ${MODEL} --loss CECWDLR --alpha_eps_ratio 0.25 --step_schedule None --restarts 1 --dataset ImageNet
    echo ${MODEL} Done
done 
