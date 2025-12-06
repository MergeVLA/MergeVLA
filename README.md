# **MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent**

**Yuxia Fu\*, Zhizhen Zhang\*, Yuqi Zhang, Zijian Wang, Zi Huang, Yadan Luo**

This repository provides the official implementation of **MergeVLA**.
**📝 [Paper](https://arxiv.org/abs/2511.18810) | 🌍 [Project Page](https://mergevla.github.io/)**

---
## :star2: Abstract
Recent Vision-Language-Action (VLA) models reformulate vision-language models by tuning them with millions of robotic demonstrations. While they perform well when fine-tuned for a single embodiment or task family, extending them to multi-skill settings remains challenging: directly merging VLA experts trained on different tasks results in near-zero success rates. This raises a fundamental question: what prevents VLAs from mastering multiple skills within one model? In this work, we identify two key sources of *non-mergeability*: (1) LoRA adapters in the VLM drift toward divergent, task-specific directions during fine-tuning, and (2) self-attention in action experts creates inter-block dependencies that prevent modular recomposition. 

MergeVLA addresses these issues with a merging-oriented architecture that preserves mergeability across tasks. It employs sparsely activated LoRA adapters via task masks to reduce irreconcilable conflicts in the VLM, and apply cross-attention-only action experts to keep specialization localized. A task router selects the appropriate mask and expert head from the initial observation to enable unsupervised task inference.

![model_arch](figures/MergeVLA_model_arch.jpg)

---

## :scroll: Table of Contents
- [:star2: Abstract](#star2-abstract)
- [:rocket: Quick Start](#rocket-quick-start)
- [:package: Data Preparation](#package-data-preparation)
- [:fire: Training](#fire-training)
- [:twisted_rightwards_arrows: Model Merging](#twisted_rightwards_arrows-model-merging)
- [:test_tube: Evaluation](#test_tube-evaluation)
- [:memo: Citation](#citation)
- [:heart: Acknowledgment](#heart-acknowledgment)

---

## :rocket: Quick Start

```bash
# Create and activate conda environment
conda create -n mergevla python=3.10.16 -y
conda activate mergevla

# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Install necessary packages
pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
pip install git+https://github.com/moojink/dlimp_openvla

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
pip install -r experiments/robot/libero/libero_requirements.txt

# Install FusionBench
cd fusion_bench
pip install -e .
```

---

## :package: Data Preparation

### LIBERO Benchmark

The LIBERO datasets can be downloaded directly from [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) or obtained following the official [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) documentation. To train on LIBERO, the raw demonstrations must be converted into the RLDS format. You may either download the RLDS-converted version from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds) or convert by yourself using this [code](https://github.com/moojink/rlds_dataset_builder).

### Performance on LIBERO benchmark. 

| Method                           | Spatial  |  Object  |   Goal   | Long-10  |   Avg    |
| -------------------------------- |:--------:|:--------:|:--------:|:--------:|:--------:|
| $\mathrm{MergeVLA}_\mathrm{FT}$  |   98.0   |   98.6   |   95.0   |   95.0   |   96.7   |
| $\mathrm{MergeVLA}_\mathrm{EMR}$ |   96.0   |   63.2   |   62.0   |   40.6   |   65.5   |
| $\mathrm{MergeVLA_{TSV}}$        | **99.4** |   97.8   |   74.4   |   54.8   |   81.6   |
| $\mathrm{MergeVLA_{KnOTS}}$      |   96.8   |   98.8   |   84.8   |   71.4   |   88.0   |
| $\mathrm{MergeVLA_{TA}}$         |   98.0   | **98.8** |   85.4   |   76.6   |   89.7   |
| $\mathrm{MergeVLA_{WUDI}}$       |   97.6   |   98.2   |   85.6   |   78.2   |   89.9   |
| $\mathrm{MergeVLA_{TIES}}$       |   94.8   |   94.6   | **91.8** | **79.4** | **90.2** |

---

## :fire: Training

Our model is trained based on the Qwen2.5-0.5B VLM, so you must download the pretrained VLM and place it under `/pretrained_models` before starting training. The training process can then be launched using the script `bash_scripts/finetune_libero.sh`. All training is performed on a single NVIDIA A6000 Ada 48GB GPU (approximately 26GB memory usage). Most task suites finish within a few hours, while the Long-10 suite requires around 24 hours. The training length is controlled by the `--max_steps` argument:

|       | Spatial | Object | Goal   | Long-10 |
| ----- | ------- | ------ | ------ | ------- |
| Steps | 30,000  | 20,000 | 30,000 | 50,000  |

Run the following command to train MergeVLA:
```bash
bash bash_scripts/finetune_libero.sh
```

---

## :twisted_rightwards_arrows: Model Merging

Model merging is implemented in `model_merging/mergy.py`. The merge algorithm is selected using `algo_name = ["TATallMask", "weighted_average"]`, where the first option merges the VLM of the model (which relies on a pretrained backbone), and the second merges the un-pretrained components, namely the action query, action head, and proprio projector, using weighted averaging by default. All available algorithms are implemented in `get_algo()`. Because merging requires access to the pretrained VLM and loading it directly is slow, we use `save_vlm()` to store the pretrained VLM inside the MergeVLA structure with zero-initialized action queries, and then use `load_vlm_from_vla()` for fast reloading during subsequent merges.

```python
if __name__ == "__main__":
    merged_tasks = ["spatial", "object", "goal", "10"]
    algo_name = ["TATallMask", "weighted_average"]
    action_head_layer_num = 1
    k_gate = 8

    merge(
        merged_tasks=merged_tasks,
        algo_name=algo_name,
        k_gate=k_gate,
        action_head_layer_num=action_head_layer_num,
        note=f'{len(merged_tasks)}tasks_AHnum_{action_head_layer_num}_k_{k_gate}'
    )
```

---

## :test_tube: Evaluation
Main evaluation script is located in `experiments/robot/libero/run_libero_eval.py`. In standard evaluation (fine-tuned Model evaluation), each task suite requires a separate checkpoint. In merged model evaluation, a single merged checkpoint can be evaluated across all task suites. In this case, we first use `task_router()` to compute the most appropriate task mask and action expert via the test-time router, and these are then applied for subsequent evaluation. The `--task_suite_name` argument is only used to load the task data; after routing, `expert_name` and `expert_idx` determine which mask and action expert to use.

```bash
# Evaluate fine-tuned Model
bash bash_scripts/eval.sh

# Evaluate merged models
bash bash_scripts/eval_merged.sh
```

---

## 📝 Citation
#### If you find this work useful in your research, please consider citing:
```bibtex
@misc{fu2025mergevla,
      title={MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent}, 
      author={Yuxia Fu and Zhizhen Zhang and Yuqi Zhang and Zijian Wang and Zi Huang and Yadan Luo},
      year={2025},
      eprint={2511.18810},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.18810}, 
}
```

---

## :heart: Acknowledgment
Our project code is built upon the following open-sourced projects: 
> [OpenvLA](https://github.com/openvla/openvla), [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter), [FusionBench](https://github.com/tanganke/fusion_bench)