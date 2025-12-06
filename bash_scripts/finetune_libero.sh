export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH

# libero_spatial_no_noops   libero_object_no_noops   libero_goal_no_noops   libero_10_no_noops
data_name=libero_spatial_no_noops
current_time=$(date "+%Y%m%d-%H%M%S")

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path /path/to/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir /path/to/modified_libero_rlds \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 50005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--wandb_entity your_wandb_entity \
--wandb_project "$data_name" \
--run_id_note MergeVLA--$data_name--$current_time