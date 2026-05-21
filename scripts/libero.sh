export PYTHONPATH=$PYTHONPATH:path_to_libero
export CUDA_VISIBLE_DEVICES=0

VLA_ID="la_direct"
LOG_DIR="logs/libero"
TASK_SUITE="libero_10"
LOG_FILE="${LOG_DIR}/${VLA_ID}-${TASK_SUITE}-1.log"

python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --vla_id $VLA_ID \
        --num_trials_per_task 50 \
        --pretrained_checkpoint path_to_pretrained_checkpoint \
        --task_suite_name $TASK_SUITE \
        --save_version $TASK_SUITE \
        --center_crop True \
        --use_pro_version True \
        --use_wandb True \
> $LOG_FILE 2>&1 &
