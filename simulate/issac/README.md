# Spider Robot Locomotion in Isaac Lab

This directory contains the Isaac Lab implementation for training an 8-legged spider robot using reinforcement learning.

## Overview

The implementation follows Isaac Lab's best practices for locomotion tasks:

- **Manager-based workflow** for modular and scalable environment design
- **MuJoCo model import** from the existing `robot/SpiderBot.xml`
- **Locomotion-specific rewards** optimized for 8-legged gait patterns
- **Curriculum learning** with progressive terrain difficulty
- **GPU-accelerated training** with thousands of parallel environments

## Directory Structure

```
simulate/issac/
├── spider_locomotion/
│   ├── __init__.py              # Environment registration
│   └── config/
│       ├── __init__.py          # Config exports
│       ├── spider_bot_cfg.py    # Robot configuration
│       └── spider_env_cfg.py    # Environment configuration
├── train_spider.py              # Training script
└── README.md                    # This file
```

## Key Features

### Robot Configuration (`spider_bot_cfg.py`)

- Imports the MuJoCo model from `robot/SpiderBot.xml`
- Configures actuators with proper stiffness and damping
- Sets initial joint positions for stable starting pose

### Environment Configuration (`spider_env_cfg.py`)

- **Observations**: Base velocities, joint positions/velocities, gravity vector, commands
- **Actions**: Joint position targets with scaling
- **Rewards**:
  - Task rewards: velocity tracking, orientation maintenance
  - Gait rewards: feet air time, alternating patterns
  - Regularization: energy efficiency, smooth motions
- **Terrain**: Progressive difficulty with curriculum learning
- **Domain Randomization**: Mass, friction, and external forces

## Prerequisites

1. **Isaac Lab Installation**: Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/)
2. **Isaac Sim 4.5**: Required version for compatibility
3. **GPU**: NVIDIA GPU with CUDA support
4. **Python 3.10**: Recommended Python version

## Training

### Method 1: Using Isaac Lab's Training Scripts (Recommended)

From the Isaac Lab root directory:

```bash
# Train with RSL_RL (recommended for quadrupeds/multi-legged robots)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless --video --video_length 500 --video_interval 1000

# Train with RL_GAMES (PPO)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless

# Train with SKRL
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless
```

### Method 2: Using the Custom Training Script

First, ensure the spider locomotion module is installed:

```bash
cd simulate/issac
pip install -e .
```

Then run:

```bash
./isaaclab.sh -p simulate/issac/train_spider.py --headless --num_envs 4096
```

### Training Parameters

- `--num_envs`: Number of parallel environments (default: 4096)
- `--headless`: Run without GUI for faster training
- `--seed`: Random seed for reproducibility
- `--max_iterations`: Maximum training iterations
- `--experiment_name`: Name for logging and checkpoints

## Evaluation

To evaluate a trained policy:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 32 \
    --checkpoint /path/to/checkpoint.pt
```

## Customization

### Modifying Rewards

Edit `spider_env_cfg.py` to adjust reward weights:

```python
@configclass
class RewardsCfg:
    # Increase velocity tracking importance
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=2.0, ...)

    # Add custom reward
    my_custom_reward = RewTerm(func=my_reward_function, weight=1.0)
```

### Changing Terrain

Modify the terrain configuration in `spider_env_cfg.py`:

```python
# For flat terrain only
terrain = TerrainImporterCfg(
    terrain_type="plane",
    # ... other settings
)
```

### Adjusting Robot Properties

In `spider_bot_cfg.py`, modify actuator properties:

```python
actuators = {
    "legs": ImplicitActuatorCfg(
        effort_limit=10.0,  # Increase max torque
        stiffness={".*": 20.0},  # Increase stiffness
        damping={".*": 1.0},  # Increase damping
    ),
}
```

## Integration with Original Training Code

This Isaac Lab implementation maintains compatibility with the original training approach while leveraging Isaac Lab's features:

1. **Curriculum Learning**: Similar 3-stage progression (balance → movement → efficiency)
2. **Reward Structure**: Adapted rewards for 8-legged locomotion
3. **Action/Observation Space**: Compatible dimensions and scaling
4. **Physics Parameters**: Matching simulation timestep and dynamics

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from within Isaac Lab environment:

   ```bash
   ./isaaclab.sh -p your_script.py
   ```

2. **CUDA Errors**: Check GPU availability and CUDA installation:

   ```bash
   nvidia-smi
   ```

3. **Performance Issues**:
   - Reduce `num_envs` if running out of GPU memory
   - Use `--headless` flag for faster training
   - Ensure GPU drivers are up-to-date

### Debug Mode

For debugging, run with fewer environments and visualization:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 64 \
    --disable_fabric
```

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab Locomotion Examples](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion)
- [Original SpiderBot Implementation](../training/)

## Next Steps

1. **Sim-to-Real Transfer**: Export trained policies for real robot deployment
2. **Advanced Gaits**: Implement specific gait patterns for different speeds
3. **Terrain Adaptation**: Add vision-based terrain recognition
4. **Multi-Task Learning**: Train for multiple objectives simultaneously
