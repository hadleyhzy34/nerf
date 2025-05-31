# PyTorch NeRF Implementation

This repository contains a PyTorch implementation of Neural Radiance Fields (NeRF), based on the paper "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Mildenhall et al. (ECCV 2020).

## Features

*   NeRF model (coarse and fine networks) with positional encoding.
*   Hierarchical volume sampling.
*   Support for the Blender dataset format.
*   Training script with:
    *   Configurable parameters (via CLI and config files using `configargparse`).
    *   Learning rate scheduling.
    *   TensorBoard logging for losses and PSNR.
    *   Checkpoint saving and resuming.
*   Evaluation script to compute PSNR, SSIM, and LPIPS metrics.
*   Rendering script to generate novel view paths and save as videos.
*   Uses `pyproject.toml` for project metadata and dependency management.

## Project Structure

```
.
├── data/                     # Placeholder for datasets (e.g., Blender dataset)
├── logs/                     # Directory for storing training logs and checkpoints
├── render_results/           # Default output for rendered novel views
├── eval_results/             # Default output for evaluation results
├── scripts/
│   ├── train_nerf.py         # Main training script
│   ├── evaluate_nerf.py      # Evaluation script
│   └── render_nerf.py        # Script for rendering novel views
├── src/
│   ├── datasets/
│   │   └── blender.py        # Blender dataset loading logic
│   ├── models/
│   │   └── nerf.py           # NeRF model, positional encoding, rendering functions
│   └── utils/
│       ├── image_utils.py    # Image processing utilities (PSNR, to_uint8)
│       └── ray_utils.py      # Ray generation utilities
├── .gitignore
├── LICENSE                   # Project License (e.g., MIT)
├── pyproject.toml            # Project metadata and dependencies
└── requirements.txt          # List of dependencies (alternative to pyproject.toml for pip install -r)
```

## Setup

**1. Clone the repository:**
```bash
git clone <repository_url>
cd <repository_name>
```

**2. Create a Python virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
You can install dependencies using `pip` with the `pyproject.toml` file or `requirements.txt`.
```bash
pip install .  # This will install the project and its dependencies from pyproject.toml
# OR
pip install -r requirements.txt
```
Ensure you have PyTorch installed according to your CUDA version if using a GPU. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

**4. Download Data:**
Download the Blender dataset (e.g., `lego`, `hotdog`, `materials`) and place it in the `data/` directory. For example:
```
data/
└── nerf_synthetic/
    ├── lego/
    │   ├── transforms_train.json
    │   ├── transforms_val.json
    │   ├── transforms_test.json
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── hotdog/
        └── ...
```
The official Blender dataset can be downloaded from the [NeRF supplementary materials website](https://www.matthewtancik.com/nerf).

## Training

Use the `scripts/train_nerf.py` script to train a NeRF model.

**Example command:**
```bash
python scripts/train_nerf.py --config configs/lego.txt --dataset_path data/nerf_synthetic/lego --expname lego_experiment
```

*   `--config`: Path to a configuration file (see below for an example).
*   `--dataset_path`: Path to the specific scene in the Blender dataset.
*   `--expname`: Name for the experiment (logs and checkpoints will be saved under `logs/{expname}`).

**Configuration File (`configs/lego.txt` example):**
Create a `configs` directory and add text files for different experiments.
```ini
# Experiment settings
expname = lego_example
basedir = ./logs/

# Dataset settings
dataset_path = data/nerf_synthetic/lego/
# data_white_bkgd = True # Use if your dataset images have alpha and you want to blend on white during loading

# Training settings
num_epochs = 200
batch_size = 1024 # Number of rays per batch
lr = 5e-4
lr_decay_factor = 0.1
lr_decay_steps = 50000 # Example, adjust based on num_epochs and dataset size

# Model settings
net_depth = 8       # Layers in network
net_width = 256     # Channels per layer
input_ch_pts_L = 10 # Positional encoding frequencies for 3D points
input_ch_views_L = 4 # Positional encoding frequencies for view directions
use_viewdirs = True

# Rendering settings for training/hierarchical sampling
N_samples_coarse = 64
N_samples_fine = 128
use_hierarchical = True
lindisp = False      # Sample linearly in depth, not disparity
white_bkgd = False   # For rendering, composite on white if true (if scene is not fully opaque)

# Logging/Saving Frequencies
log_freq = 100       # Log to TensorBoard every N iterations
save_freq = 10       # Save checkpoint every N epochs
val_img_freq = 5000 # Render a validation image every N iterations (placeholder)

# Other
seed = 42
num_workers = 4      # DataLoader workers
# resume_from_checkpoint = latest.pth # or path/to/specific.pth
```
**Note:** The `val_img_freq` currently corresponds to a placeholder in the training script.

**To view training progress with TensorBoard:**
```bash
tensorboard --logdir ./logs
```
Navigate to `http://localhost:6006` in your browser.

## Evaluation

Use `scripts/evaluate_nerf.py` to evaluate a trained model.

**Example command:**
```bash
python scripts/evaluate_nerf.py \
    --checkpoint_path logs/lego_experiment/checkpoints/checkpoint_final_epoch_0199_step_XXXXXXXX.pth \
    --dataset_path data/nerf_synthetic/lego \
    --dataset_split test \
    --output_dir eval_results/lego_experiment_test
```
*   `--checkpoint_path`: Path to the trained model checkpoint (`.pth` file).
*   `--dataset_path`: Path to the dataset.
*   `--dataset_split`: Which split to evaluate ('test' or 'val').
*   `--output_dir`: Directory to save rendered images and `metrics_summary.json`.

The script will output mean PSNR, SSIM, and LPIPS metrics and save individual rendered images.

## Rendering Novel Views

Use `scripts/render_nerf.py` to generate a video of novel views from a trained model.

**Example command:**
```bash
python scripts/render_nerf.py \
    --checkpoint_path logs/lego_experiment/checkpoints/checkpoint_final_epoch_0199_step_XXXXXXXX.pth \
    --output_dir render_results/lego_experiment_spiral \
    --render_path_type spherical_spiral \
    --num_frames 120 \
    --render_H 400 --render_W 400 --render_focal 555.55 # Adjust H,W,focal as needed or rely on checkpoint
```
*   `--checkpoint_path`: Path to the trained model checkpoint.
*   `--output_dir`: Directory to save video frames and the final `.mp4` video.
*   `--render_path_type`: Type of camera path (e.g., 'spherical_spiral', 'circle').
*   `--num_frames`: Number of frames for the video.
*   `--render_H, --render_W, --render_focal`: Optional overrides for rendering dimensions/focal. If not provided, the script attempts to use values from the checkpoint or defaults.

## TODO / Future Work

*   Implement validation image rendering during training.
*   Add more sophisticated camera path options for rendering.
*   Support for other datasets (e.g., LLFF).
*   Further code cleanup, full docstring coverage, and unit tests.
*   Integration of linters/formatters like Black and Ruff more formally.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
(Assuming MIT based on the `pyproject.toml` entry. Update if different.)

## Acknowledgements

*   Based on the original NeRF paper by Mildenhall et al.
*   Inspired by various public NeRF implementations.
```
