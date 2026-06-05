# Spider Robot Locomotion in Genesis

This directory contains the Genesis implementation for training an 8-legged spider robot using reinforcement learning.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies

```bash
uv sync
```

## Run training

```bash
uv run python -m spiderbot.locomotion.train
```

To run more parallel training environments on mixed terrain:

```bash
uv run python -m spiderbot.locomotion.train -t mixed -n 4048
```

## Play 

You can play the trained agent with a Logitech gamepad.

```bash
uv run python -m spiderbot.locomotion.play ./logs/<TRAINING DIR>
```

## Without uv (pip/python)

Install dependencies

```bash
pip install -r requirements.txt
```

## Run training

```bash
python -m spiderbot.locomotion.train
```

## Play 

You can play the trained agent with a Logitech gamepad.

```bash
python -m spiderbot.locomotion.play ./logs/<TRAINING DIR>
```
