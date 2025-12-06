export PYTHONPATH=/path/to/MergeVLA:$PYTHONPATH
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH

num_trial=50
ckpt=/path/to/ckpt
tasks=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for task in "${tasks[@]}"; do
  python experiments/robot/libero/run_libero_eval.py \
    --num_images_in_input 2 \
    --pretrained_checkpoint $ckpt \
    --task_suite_name $task \
    --load_moe True \
    --pretrained_vlm_checkpoint /path/to/Pretrained-VLM \
    --k_gate 8 \
    --action_head_layer_num 1 \
    --num_trials_per_task $num_trial \
done