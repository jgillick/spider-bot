# Isaac Lab Installation Guide

This guide explains how to install the Spider Locomotion task in Isaac Lab so you can use **Method 1** from the README (using Isaac Lab's built-in training scripts).

## Prerequisites

1. **Isaac Lab installed** in your cloud environment
2. **ISAACLAB_ROOT environment variable** set to your Isaac Lab installation directory
3. **Python 3.10+** with pip

## Quick Setup (Recommended)

### Step 1: Set Environment Variable

```bash
# Set this to your Isaac Lab installation directory
export ISAACLAB_ROOT=/path/to/your/IsaacLab

# Verify Isaac Lab is found
ls $ISAACLAB_ROOT
```

### Step 2: Run Setup Script

```bash
# Navigate to the spider locomotion directory
cd /path/to/SpiderBot/simulate/issac

# Run the automated setup
./setup_cloud.sh
```

The setup script will:

- Install the spider locomotion package
- Copy files to Isaac Lab's task directory
- Test the installation
- Provide usage instructions

## Manual Setup (Alternative)

If the automated setup doesn't work, follow these manual steps:

### Method A: Package Installation

```bash
# Navigate to spider locomotion directory
cd /path/to/SpiderBot/simulate/issac

# Install in development mode
pip install -e .

# Test installation
python test_environment.py
```

### Method B: Direct File Copy

```bash
# Set Isaac Lab root
export ISAACLAB_ROOT=/path/to/your/IsaacLab

# Create task directory
mkdir -p $ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion

# Copy spider locomotion files
cp -r /path/to/SpiderBot/simulate/issac/spider_locomotion/* \
      $ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion/

# Copy robot XML file and meshes
mkdir -p $ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion/assets
cp /path/to/SpiderBot/simulate/robot/SpiderBotNoEnv.xml \
   $ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion/assets/

# Copy the meshes directory (required for XML references)
cp -r /path/to/SpiderBot/simulate/robot/meshes \
   $ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion/assets/
```

## Verification

After installation, verify the task is registered:

```bash
# Test environment registration
cd /path/to/SpiderBot/simulate/issac
python test_environment.py
```

You should see:

```
✓ Spider locomotion module imported successfully
✓ Isaac-SpiderLocomotion-v0 is registered
✓ Isaac-SpiderLocomotion-Flat-v0 is registered
✓ Configuration classes imported successfully
✓ MuJoCo model found at: /path/to/SpiderBot/simulate/robot/SpiderBotNoEnv.xml
✅ Environment setup verified! Ready for training in Isaac Lab.
```

## Usage

Once installed, you can use **Method 1** from the README:

### Training with RSL_RL (Recommended)

```bash
# From Isaac Lab root directory
cd $ISAACLAB_ROOT

# Train with rough terrain
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless

# Train with flat terrain (easier to start)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-SpiderLocomotion-Flat-v0 \
    --num_envs 4096 \
    --headless
```

### Training with RL_GAMES (PPO)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless
```

### Training with SKRL

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-SpiderLocomotion-v0 \
    --num_envs 4096 \
    --headless
```

## Troubleshooting

### Issue: "Task not found" Error

If you get an error like `Task 'Isaac-SpiderLocomotion-v0' not found`:

1. **Check installation**:

   ```bash
   python test_environment.py
   ```

2. **Verify Isaac Lab environment**:

   ```bash
   ./isaaclab.sh -p -c "import gymnasium; print(gymnasium.envs.registry.keys())"
   ```

3. **Reinstall the package**:
   ```bash
   cd /path/to/SpiderBot/simulate/issac
   pip uninstall spider_locomotion_isaaclab -y
   pip install -e .
   ```

### Issue: XML File Not Found

If you get an error about the XML file:

1. **Check file exists**:

   ```bash
   ls -la /path/to/SpiderBot/simulate/robot/SpiderBotNoEnv.xml
   ```

2. **Update path in configuration**:
   Edit `spider_locomotion/config/spider_bot_cfg.py` and update the path.

### Issue: Import Errors

If you get import errors:

1. **Check Isaac Lab installation**:

   ```bash
   ./isaaclab.sh -p -c "import isaaclab; print('Isaac Lab OK')"
   ```

2. **Verify Python environment**:
   ```bash
   ./isaaclab.sh -p -c "import sys; print(sys.path)"
   ```

## File Structure After Installation

Your Isaac Lab installation should have this structure:

```
$ISAACLAB_ROOT/
├── source/
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/
│           └── manager_based/
│               └── locomotion/
│                   └── spider_locomotion/
│                       ├── __init__.py
│                       ├── config/
│                       │   ├── __init__.py
│                       │   ├── spider_bot_cfg.py
│                       │   ├── spider_env_cfg.py
│                       │   └── spider_flat_env_cfg.py
│                       └── assets/
│                           ├── SpiderBotNoEnv.xml
│                           └── meshes/
│                               ├── Body.stl
│                               ├── Leg1_Femur-actuator-assembly_Femur.stl
│                               ├── Leg1_Hip-actuator-assembly_Motor.stl
│                               └── ... (all other .stl files)
└── scripts/
    └── reinforcement_learning/
        ├── rsl_rl/
        ├── rl_games/
        └── skrl/
```

## Performance Tips for Cloud

1. **Use headless mode** for faster training:

   ```bash
   --headless
   ```

2. **Adjust number of environments** based on GPU memory:

   ```bash
   --num_envs 2048  # Reduce if out of GPU memory
   ```

3. **Monitor GPU usage**:

   ```bash
   nvidia-smi
   ```

4. **Use tensorboard for monitoring**:
   ```bash
   tensorboard --logdir runs/
   ```

## Next Steps

After successful installation:

1. **Start with flat terrain** to verify everything works
2. **Monitor training progress** with tensorboard
3. **Adjust hyperparameters** in the configuration files
4. **Export trained policies** for deployment

## Support

If you encounter issues:

1. Check the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/)
2. Review the troubleshooting section above
3. Check the original README.md for detailed configuration options
4. Verify your Isaac Lab version is compatible (4.5+ recommended)
