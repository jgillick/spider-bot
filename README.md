# The Spider Bot

A spider robot which uses reinforcement learning to walk

See the project log: https://hackaday.io/project/202366-the-unnamed-spider-bot

## Installation

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and dependencies automatically using `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

## Train Model

```bash
cd ./spider-bot
python -m spiderbot.locomotion.train
```

## Motor Sysid

Identify real motor parameters (friction, inertia, gains) from a connected ODrive-compatible controller and generate copy-paste values for the sim environment. See [hardware/README.md](hardware/README.md) for full instructions.

```bash
uv run python -m spiderbot.hardware.sysid --gear-ratio 8
```

The interactive menu includes encoder-frame verification (option 1) and all identification experiments.
