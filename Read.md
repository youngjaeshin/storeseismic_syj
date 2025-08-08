# Read.md

## Overview
This repository provides utilities to generate synthetic seismic data and train BERT-based models for tasks such as denoising or time-picking. Each workflow is controlled through YAML configuration files so experiments are reproducible and easy to modify.

## Getting Started
1. Prepare a Python 3.10+ environment.
2. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio tqdm pyyaml matplotlib numpy
   ```
   If a GPU is available, install the CUDA-enabled PyTorch wheel for faster training.

## Generate Synthetic Data
The repository includes scripts that build random velocity models and corresponding seismograms.
1. **Velocity model generation**
   ```bash
   python data_generation/generate_velocity_models.py --config configs/data_generation.yaml
   ```
   Edit `configs/data_generation.yaml` to change the number of models, velocity ranges or output directory.
2. **Seismogram generation**
   ```bash
   python data_generation/generate_seismograms.py --config configs/data_generation.yaml
   ```
   Generated models and traces will appear under `synthetic_data/raw_generated/`.

## Train the Model
`main.py` reads configuration files that describe the training task.
- **Pre-training example**
  ```bash
  python main.py --config configs/pretrain_time.yaml
  ```
- **Fine-tuning example (denoising)**
  ```bash
  python main.py --config configs/finetune_denoising_time.yaml
  ```
You can supply any other configuration file from the `configs` directory to run different experiments. Results, model weights and plots are written to `<base_results_dir>/<run_name>/` as defined in each config.

## Inspect Results
After training, check the run directory for:
- `*.pt` files containing the trained weights
- `training_curve.png` for loss curves
- Additional artifacts produced by your task

## Run Tests
After modifying the code, run the unit tests:
```bash
pytest
```
The tests require all dependencies (including PyTorch) to be installed.

## Questions or Issues?
Feel free to open an issue if you encounter problems or have improvement suggestions.
