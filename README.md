<div align="center">
  <h1>From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models</h1>
  <p>
    <strong>Language / 语言:</strong> <strong>English</strong> | <a href="./README_zh.md">中文</a>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2605.04678">✍️ <strong>Paper (arXiv)</strong></a> |
    <a href="https://github.com/RUCKBReasoning/From_Pixels_to_Tokens">💻 <strong>Code</strong></a> |
    <a href="https://huggingface.co/CokeAnd1ce/From_Pixels_to_Tokens">🤗 <strong>Checkpoints, Dataset</strong></a> |
  </p>
</div>

<p align="center">
  <img src="asserts/Figure_1.png" alt="Overview" width="100%">
</p>

## ✨ News ✨

- [2026/05/26] Add LIBERO datasets and evaluation code.
- [2026/05/24] Our paper was accepted as an ICML 2026 Oral✨.
- [2026/05/06] Our paper is now available on arXiv. We also release the project code, checkpoints, and dataset.
- [2026/05/01] Our paper was accepted as an ICML 2026 Spotlight.

## Overview

This work studies how latent action supervision can be integrated into Vision-Language-Action (VLA) models under a unified training framework. We focuses on how different latent supervision strategies change downstream VLA policy learning.

Our implementation is built on a shared `Qwen3-VL-2B` backbone and compares four representative strategies:

| Model       | Latent supervision          | Role in VLA training                                      |
| ----------- | --------------------------- | --------------------------------------------------------- |
| `Baseline`  | None                        | Direct action prediction without latent supervision       |
| `LA-Align`  | Image-based latent actions  | Align internal VLM representations with latent embeddings |
| `LA-Direct` | Image-based latent actions  | Directly decode latent actions as discrete tokens         |
| `LA-Cond`   | Image-based latent actions  | Jointly decode latent actions and action representations  |
| `LA-Tok`    | Action-based latent actions | Map actions into discrete latent tokens                   |

This project follows two complementary perspectives from the paper:

- Image-based latent actions for trajectory regularization
- Action-based latent actions for target-space unification

## 🧩 Method

<p align="center">
  <img src="asserts/Figure_2.png" alt="Architecture" width="100%">
</p>

All methods share the same VLA backbone and action head, and differ only in how latent supervision is injected. The main VLA implementations live in [`latentvla/models/vla`](./latentvla/models/vla).

## 📦 Installation

```bash
conda create -n latentvla python=3.10 -y
conda activate latentvla

pip install --upgrade pip setuptools wheel
```

Install the PyTorch 2.8 stack. The command below uses CUDA 12.8 wheels:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the project requirements from the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Install `dlimp`, which is used by the RLDS data pipeline:

```bash
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla.git
```

Install FlashAttention 2. The VLA training path uses
`attn_implementation="flash_attention_2"`, or can use `attn_implementation="sdpa"`.

```bash
pip install packaging ninja
ninja --version
pip install flash-attn==2.8.3 --no-build-isolation
```

If you already have a compatible prebuilt FlashAttention wheel, you can install
it directly instead:

```bash
pip install /path/to/flash_attn-2.8.3-*.whl
```

## 💾 Data Preparation

The repository assumes RLDS-style datasets for both latent action preprocessing and VLA training.

Public RLDS-format datasets used in this project include:

- [LIBERO](https://huggingface.co/CokeAnd1ce/From_Pixels_to_Tokens)
- [RoboTwin 2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- [JAKA dataset](https://huggingface.co/CokeAnd1ce/From_Pixels_to_Tokens)

After downloading and preparing the dataset locally, point `data_root_dir` to the dataset root.

## ⚙️ Training Latent Action Models

### A. Image-based latent action model

The image-based latent action model is in [`data_preprocess/image_based_lam`](./data_preprocess/image_based_lam). It follows a UniVLA-style image-based latent action pipeline and is the part used to produce latent supervision for `LA-Align`, `LA-Direct`, and `LA-Cond`.

Before post-training on your dataset, download the two public [UniVLA(RSS2025) checkpoints](https://github.com/OpenDriveLab/UniVLA):

- Stage-1 checkpoint
- Stage-2 checkpoint

These checkpoints are used as initialization because it performs dataset-specific post-training rather than training the image-based latent model entirely from scratch.

#### Training

Edit [`data_preprocess/image_based_lam/config/lam-stage-2.yaml`](./data_preprocess/image_based_lam/config/lam-stage-2.yaml):

- `model.lam_path`
- `model.stage_one_ckpt`
- `data.data_root_dir`
- `data.data_mix`
- `trainer.devices`
- logging and checkpoint paths

Then run:

```bash
cd data_preprocess/image_based_lam
torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
  --config config/lam-stage-2.yaml
```

### B. Annotate RLDS data with image-based latent labels

After training the image-based model, use [`data_preprocess/image_based_lam/latent.py`](./data_preprocess/image_based_lam/latent.py) to annotate trajectories with latent labels. The current script gives a `LIBERO`-style TFRecord example and writes:

- `steps/latent_idx`
- `steps/latent_z`

Before running it, edit the checkpoint path inside the script:

```python
lam_path = "your_lam_checkpoint.pth"
```

Then run:

```bash
cd data_preprocess/image_based_lam
python latent.py /path/to/file.tfrecord
```

The script writes a new TFRecord file into a sibling `output/` directory next to the input file.

### C. Action-based latent action model

The action-based latent action model is in [`data_preprocess/action_based_lam`](./data_preprocess/action_based_lam). This one is trained from scratch in this repository and is used by `LA-Tok`.

The tokenizer learns a VQ-style discrete latent space over action chunks and saves checkpoints of the form `tokenizer_step_*.pt`.

You can launch training with [`data_preprocess/action_based_lam/action.sh`](./data_preprocess/action_based_lam/action.sh) after editing:

- `--data-root-dir`
- `--data-mix`
- `--results-dir`
- `--num-steps`

Run:

```bash
cd data_preprocess/action_based_lam
bash action.sh
```

## 🚀 Training

The main training entry is [`exp/train_vla.py`](./exp/train_vla.py).

Before VLA training, first download the `Qwen3-VL-2B` [checkpoint](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and set:

```bash
--vlm_path /path/to/Qwen3-VL-2B
```

For the simulation benchmark, checkpoints are available on Hugging Face: [simulation benchmark ckpts](https://huggingface.co/CokeAnd1ce/From_Pixels_to_Tokens)

Supported `--vla_id` values:

- `baseline`
- `la_align`
- `la_direct`
- `la_cond`
- `la_tok`

Before launching training, make sure you set:

- `--vlm_path` to the downloaded `Qwen3-VL-2B` checkpoint
- `--data_root_dir` to the RLDS dataset root
- `--data_mix` to the target dataset split or mixture
- `--action_tokenizer_ckpt` when training `la_tok`
- `--pretrained_checkpoint` and `--from_pretrained True` when loading a checkpoint for continued training or evaluation

Example baseline command:

```bash
torchrun --nnodes=1 --nproc_per_node=1 exp/train_vla.py \
  --seed 42 \
  --run_root_dir runs \
  --save_checkpoint True \
  --vla_id baseline \
  --vlm_path /path/to/Qwen3-VL-2B \
  --vlm_model_id Qwen3 \
  --default_image_size 224 \
  --data_root_dir /path/to/rlds_data \
  --data_mix '["libero_goal"]' \
  --shuffle_buffer_size 128 \
  --image_aug True \
  --window_size 8 \
  --use_wrist_image True \
  --use_proprio True \
  --type training \
  --epochs 10 \
  --max_steps 20000 \
  --global_batch_size 128 \
  --per_device_batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.03 \
  --save_step 20000 \
  --wandb_project your_project \
  --use_wandb True
```

For the other variants, change `--vla_id`:

```bash
--vla_id la_align
--vla_id la_direct
--vla_id la_cond
--vla_id la_tok
```

For `la_tok`, also add:

```bash
--action_tokenizer_ckpt /path/to/tokenizer_step_xxxxx.pt
```

## 🧪 Evaluation

For LIBERO evaluation, we provide a ready-to-run script: [`scripts/libero.sh`](./scripts/libero.sh).

Before running it, edit the following placeholders in the script:

- `PYTHONPATH` -> `path_to_libero`
- `--pretrained_checkpoint` -> `path_to_pretrained_checkpoint`
- `VLA_ID` -> one of `la_align`, `la_direct`, `la_cond`, `la_tok`, `baseline`

Then launch evaluation with:

```bash
bash scripts/libero.sh
```

The script runs [`experiments/robot/libero/run_libero_eval.py`](./experiments/robot/libero/run_libero_eval.py) in the background and writes logs to `logs/libero/`.

## 📝 Notes

- Robot-specific constants are selected in [`latentvla/models/constants.py`](./latentvla/models/constants.py) by parsing command-line arguments. If your dataset name does not clearly indicate the robot platform, adjust that file manually.
- The codebase expects RLDS-format training data.
- Some preprocessing scripts still contain placeholder paths and should be edited before first use.
- `swanlab` logging is opt-in. It will only initialize when you explicitly set `ENABLE_SWANLAB=1`.

## 🙏 Acknowledgements

We thank [OpenVLA](https://github.com/openvla/openvla), [UniVLA](https://github.com/OpenDriveLab/UniVLA), [StarVLA](https://github.com/starVLA/starVLA), [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) and [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) for their open-sourced work!

## 📚 BibTeX

```bibtex
@article{pixels2tokens2026,
  title   = {From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models},
  author  = {Lin, Yihan and Li, Haoyang and Li, Yang and Shen, Haitao and Zhao, Yihan and Shao, Chao and Zhang, Jing},
  journal = {arXiv preprint arXiv:2605.04678},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.04678}
}
```
