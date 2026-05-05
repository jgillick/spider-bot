# The Spider Bot

A spider robot which uses reinforcement learning to walk

See the project log: https://hackaday.io/project/202366-the-unnamed-spider-bot

## Installation

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and dependencies automatically using `pyproject.toml` and `uv.lock`.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a local .venv
uv sync
```

Run any script through uv so it always uses the managed environment:

```bash
uv run python spider-bot/train_rsl.py
uv run python -m hardware.sysid --gear-ratio 8
```

Add a new dependency:

```bash
uv add some-package
```

This updates both `pyproject.toml` and `uv.lock`. Commit both files.

### With pip (alternative)

Create a Python 3.11 virtual environment and install dependencies manually:

```bash
pyenv virtualenv 3.11 spider-bot
pyenv activate spider-bot
pip install --upgrade pip
pip install -r requirements.txt
```

## Train Model

```bash
cd ./spider-bot
uv run python train_rsl.py
```

## Motor Sysid

Identify real motor parameters (friction, inertia, gains) from a connected ODrive-compatible controller and generate copy-paste values for the sim environment. See [hardware/README.md](hardware/README.md) for full instructions.

```bash
# Verify encoder frame of reference (run once per motor)
uv run python hardware/verify_encoder.py --gear-ratio 8

# Run identification experiments
uv run python -m hardware.sysid --gear-ratio 8
```
