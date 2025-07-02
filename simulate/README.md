# 🕷️ Spider Robot Reinforcement Learning Environment

This package contains a custom Gymnasium environment for training an 8-legged spider robot using torque control in MuJoCo and Stable Baselines3.

---

## 📦 Installation

### 1. Create a Python Virtual Environment

```bash
pyenv virtualenv 3.11.9 spider-bot
pyenv activate spider-bot
```

### 2. Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Running Training

Make sure your working directory contains the `SpiderBody.xml`.

```bash
python -m training.train_spider
```

---

## 🧪 Debugging

To test the environment manually:

```python
from training.spider_torque_env import SpiderTorqueEnv
env = SpiderTorqueEnv("SpiderBody.xml", render_mode="human")
obs, _ = env.reset()
done = False
while not done:
    obs, reward, done, _, _ = env.step(env.action_space.sample())
    env.render()
env.close()
```

---

## 📊 Monitoring Progress

```bash
tensorboard --logdir logs/
```

---

## 📹 Video Playback

Videos are saved every 10k steps in:

```
logs/videos/
```

---

## 🧠 Based on

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [MuJoCo](https://mujoco.readthedocs.io/)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

Happy training!
