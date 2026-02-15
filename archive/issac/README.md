# Spider Robot Locomotion in Isaac Lab

This directory contains the Isaac Lab implementation for training an 8-legged spider robot using reinforcement learning.

## Installation

### Local requirements

Create a Python 3.11 virtual environment.

```bash
pyenv virtualenv 3.11 spider-bot
pyenv activate spider-bot
```

Install dependencies

````bash
pip install --upgrade pip
pip install -r requirements.txt
````

### Isaac Lab/Sim

Install [Isaac Lab](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html)

Set the Isaac Lab installation path in `env.cfg`:

```bash
# Copy and then edit file
cp env.cfg.example env.cfg
```

### Install training program into Isaac Lab

```bash
./install.sh
```


## Run training

```bash
./train.sh
```