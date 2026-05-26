export PYTHONPATH=$PYTHONPATH:path_to_libero
export CUDA_VISIBLE_DEVICES=0

VLA_ID="la_direct"
ACTION_HEAD_TYPE="flow_gr00t"
FLOW_DIT_SIZE="dit-b"
CODEBOOK_SIZE="8"
LATENT_TOKENS_PER_STEP="16"
CKPT_PATH="path_to_pretrained_checkpoint"
VLM_DIR="/ssd/linyihan/ckpt/Qwen3-VL-2B-Instruct"
LOG_DIR="logs/libero"
TASK_SUITE="libero_10"
LOG_FILE="${LOG_DIR}/${VLA_ID}-${ACTION_HEAD_TYPE}-${FLOW_DIT_SIZE}-${TASK_SUITE}.log"

mkdir -p "$LOG_DIR"

python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --vla_id $VLA_ID \
        --action_head_type $ACTION_HEAD_TYPE \
        --flow_dit_size $FLOW_DIT_SIZE \
        --codebook_size $CODEBOOK_SIZE \
        --latent_tokens_per_step $LATENT_TOKENS_PER_STEP \
        --num_trials_per_task 50 \
        --pretrained_checkpoint $CKPT_PATH \
        --vlm_model_dir $VLM_DIR \
        --task_suite_name $TASK_SUITE \
        --save_version $TASK_SUITE \
        --center_crop True \
        --use_proprio True \
        --use_pro_version True \
        --use_wandb True \
> $LOG_FILE 2>&1 &
