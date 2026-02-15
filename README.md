# The Spider Bot

A spider robot which uses reinforcement learning to walk

See the project log: https://hackaday.io/project/202366-the-unnamed-spider-bot

## Installation

### Local requirements

Create a Python 3.11 virtual environment.

```bash
pyenv virtualenv 3.11 spider-bot
pyenv activate spider-bot
```

Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Train Model

```bash
cd ./spider-bot
python ./train_rsl.py
```
