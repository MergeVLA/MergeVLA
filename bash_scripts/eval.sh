export PYTHONPATH=/path/to/MergeVLA:$PYTHONPATH
export PYTHONPATH=/path/to/LIBERO:$PYTHONPATH

ckpt=/path/to/ckpt
task=libero_spatial # libero_spatial libero_object libero_goal libero_10

python experiments/robot/libero/run_libero_eval.py \
  --num_images_in_input 2 \
  --pretrained_checkpoint $ckpt \
  --task_suite_name $task \
  --load_moe False \
  --num_trials_per_task 50