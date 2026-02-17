# Spider Robot RL Training with Gymnasium & Stable Baselines 3

This directory contains the reinforcement learning environment and training scripts for the spider robot with
gymnasium and stable baselines 3.

## Contents

- `environment.py` - Simplified Gymnasium environment with curriculum learning
- `train.py` - Comprehensive training script with multiple configurations

## 📦 Installation

### 1. Create a Python Virtual Environment

```bash
# Create
pyenv virtualenv 3.11.9 spider-bot

# Activate
pyenv activate spider-bot
```

### 2. Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Running Training

```bash
python train <algorithm>

# Train with the PPO training algorithm
python train PPO

# Train with the SAC training algorithm
python train SAC
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=./logs
```

### Videos

Check `out/videos/` for recorded episodes showing robot behavior.

### Logs

Training progress is printed to console and logged to TensorBoard.
