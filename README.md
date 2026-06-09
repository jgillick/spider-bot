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

## Autonomous Jumping Agent

An AI agent that autonomously iterates on reward functions to teach the robot to jump. Uses Claude to propose modifications, runs probe training sessions, evaluates jump quality, and loops until the robot achieves a clean forward jump.

See [spiderbot/jumping/README.md](spiderbot/jumping/README.md) for full setup and usage instructions.

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
python -m spiderbot.jumping.run_agent
```

## Motor Sysid

Identify real motor parameters (friction, inertia, gains) from a connected ODrive-compatible controller and generate copy-paste values for the sim environment. See [hardware/README.md](hardware/README.md) for full instructions.

```bash
uv run python -m spiderbot.hardware.sysid --gear-ratio 8
```

The interactive menu includes encoder-frame verification (option 1) and all identification experiments.
