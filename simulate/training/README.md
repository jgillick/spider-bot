# Spider Robot Reinforcement Learning Training

This directory contains the reinforcement learning environment and training scripts for the spider robot.

## Contents

- `environment.py` - Simplified Gymnasium environment with curriculum learning
- `train.py` - **CONSOLIDATED**: Comprehensive training script with multiple configurations
- `test.py` - Script for testing trained models

## Quick Start

### Using the Consolidated Training System (Recommended)

```bash
# Run the interactive training script
python -m training.train

# Or run directly
python train.py

# Or with custom config
python -c "
from training.train import train_spider_improved
config = {
    'num_envs': 8,
    'total_timesteps': 800_000,
    'early_stopping': True,
    'ent_coef': 0.05,  # Increased exploration
    'generate_videos': True
}
train_spider_improved('../robot/SpiderBot.xml', config)
"
```

## Training Configurations

The consolidated `train.py` script offers three training configurations:

### 1. Basic Training

- **Purpose**: Simple training with standard parameters
- **Best for**: Quick experiments and baseline testing
- **Features**: 4 parallel environments, standard exploration, basic curriculum

### 2. Improved Training (Default)

- **Purpose**: Enhanced training with curriculum advancement and anti-overfitting
- **Best for**: Most training scenarios
- **Features**:
  - 8 parallel environments
  - Curriculum advancement on plateau
  - Increased exploration (ent_coef=0.05)
  - Reduced clipping for stability
  - More frequent evaluation

### 3. Walking Training

- **Purpose**: Optimized specifically for learning to walk
- **Best for**: When the robot needs to master walking behavior
- **Features**:
  - Higher stage thresholds (13k, 15k, 18k)
  - More patience before advancing (10 evaluations)
  - Standard exploration parameters
  - Longer training between evaluations

## System Comparison

The training system has been analyzed and improved. Key findings:

- **Original System**: Over-optimized reward function (12+ components) but under-optimized infrastructure
- **Improved System**: Simplified rewards with curriculum learning and parallel training

## Requirements

```bash
pip install -r ../requirements.txt
```

## Key Features

### Original Environment

- 87-dimensional observation space
- Complex reward with 12+ components
- Single environment training
- No curriculum learning

### Enhanced Training System

- **48-dimensional observation space** (simplified from 87)
- **Staged curriculum learning** (Balance → Movement → Efficiency)
- **Parallel training** (4-8 environments for 4-8x speedup)
- **Automatic video generation** after training completes
- **Adaptive learning rate** and comprehensive checkpointing
- **Real-time monitoring** with tensorboard integration
- **🛡️ Anti-overfitting features** (early stopping, increased exploration)
- **📊 Better evaluation** with manual evaluation to avoid deadlocks

### Video Generation Features

- **Automatic creation** of training videos after model completion
- **Multiple episodes** recorded showing learned walking behavior
- **Demo videos** with different camera perspectives
- **High-quality MP4 output** for easy sharing and analysis
- **Configurable video settings** (length, frequency, quality)
- **MoviePy 2.x compatibility** with graceful fallback

## Training Tips

1. **Start Simple**: Use the basic configuration for initial testing
2. **Use Improved**: Switch to improved configuration for most training
3. **Monitor Progress**: Watch tensorboard for stage progression
4. **Adjust Thresholds**: Modify stage thresholds based on robot performance
5. **Parallel Training**: Use 4-8 environments for 4-8x speedup
6. **Prevent Overfitting**: Use improved configuration with early stopping
7. **Monitor Evaluation**: Watch for reward decreases indicating overfitting

## Anti-Overfitting Features

The improved training system includes several features to prevent overfitting:

### Early Stopping

- **Automatic detection** of performance degradation
- **Configurable patience** (default: 5 evaluations)
- **Minimum improvement threshold** (default: 1.0 reward)
- **Best model preservation** at peak performance

### Increased Exploration

- **Higher entropy coefficient** (0.05 vs 0.01)
- **Reduced clipping range** (0.1 vs 0.2)
- **More frequent evaluation** (every 25k vs 50k steps)

### Better Monitoring

- **Manual evaluation** to avoid deadlocks
- **Detailed logging** of improvement tracking
- **Evaluation history** for trend analysis
- **Best performance tracking** with timestamps

## Curriculum Advancement on Plateau

The consolidated training script includes:

### Smart Curriculum Progression

- **Dual progression triggers**:
  - Performance threshold exceeded (natural progression)
  - Learning plateau detected (forced progression)
- **Stage-specific metrics**: Tracks best performance per stage
- **Automatic environment updates**: Updates both training and eval environments
- **Metric reset**: Resets evaluation metrics for each new stage

### Benefits

- **No wasted training**: Advances when learning stops improving
- **Complete curriculum**: Ensures all stages are trained
- **Optimal models**: Saves best model for each stage
- **Efficient training**: Stops only when final stage plateaus

## Troubleshooting

- **Robot falls immediately**: Check XML model and initial pose
- **No movement**: Curriculum may be progressing too fast, increase thresholds
- **Jerky motion**: Reduce max torque or increase damping
- **Slow training**: Increase number of parallel environments
- **Training freezes during evaluation**: This was a known issue with `evaluate_policy` that has been fixed by using manual evaluation instead
- **Overfitting (rewards decrease after peak)**: Use improved configuration with early stopping and increased exploration
- **No curriculum progression**: Lower stage thresholds in config
- **MoviePy import error**: Install moviepy 2.x with `pip install moviepy>=2.0.0`

## Next Steps

After training, you can:

1. Visualize the learned behavior using `test.py`
2. Export the policy for deployment
3. Fine-tune with different reward weights
4. Try domain randomization for robustness

## File Structure

- **`environment.py`** - The main environment class (`SpiderRobotEnv`)
- **`train.py`** - **CONSOLIDATED**: Training script with multiple configurations
- **`test.py`** - Testing and evaluation script for trained models

## Environment Details

### Action Space

- **24 joint positions** (8 legs × 3 joints each)
- Uses PD control with Kp=10.0, Kd=0.5
- Torque limits: ±8.0 N⋅m

### Observation Space (48 dimensions - simplified)

- Joint positions (24)
- Joint velocities (24)

### Reward Function

- Forward progress (moderate weight)
- Height maintenance (critical)
- Upright orientation (critical)
- Body stability
- Gait coordination
- Weight distribution
- Energy efficiency
- Survival bonus

## Weight Distribution Feature

The environment now includes a weight distribution reward that encourages the robot to distribute its weight evenly across contacting feet. This improves stability and prevents overloading individual legs.

### How it works:

1. **Contact Force Measurement**: Measures actual contact forces on each foot
2. **Distribution Analysis**: Calculates coefficient of variation (CV) of forces
3. **Lateral Balance**: Ensures left and right sides carry similar loads
4. **Reward Structure**:
   - 70% weight for even distribution across all contacting feet
   - 30% weight for left-right balance

### Benefits:

- **Improved Stability**: Prevents tipping and wobbling
- **Natural Gait**: Encourages more realistic walking patterns
- **Load Sharing**: Reduces stress on individual legs
- **Better Sim-to-Real Transfer**: More realistic physics

### Metrics Tracked:

- `weight_distribution_cv`: Coefficient of variation (lower = better)
- `lateral_balance`: Left-right balance ratio (1.0 = perfect balance)
- `num_feet_in_contact`: Number of feet touching the ground

## Hardware Deployment

When deploying to real ODrive 3.6 controllers:

### Control Mode: MIT (Motor, Impedance, Torque)

For each of the 24 actuators, send:

- **Position**: Target joint angle (radians)
- **Velocity**: 0 (static targets)
- **Kp**: 10.0 (position gain)
- **Kd**: 0.5 (velocity/damping gain)
- **Torque_ff**: 0 (no feedforward)

### Safety Considerations

1. Implement emergency stop capability
2. Monitor joint limits and torque limits
3. Test in simulation first
4. Start with low gains and gradually increase
5. Have manual override capability

## Training Parameters

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **Network**: [256, 256, 128] for both policy and value function
- **Entropy Coefficient**: 0.01 (for exploration)
- **Gamma**: 0.99 (discount factor)

## Troubleshooting

### Common Issues

1. **Robot falls immediately**:

   - Check joint limits in the XML file
   - Reduce control gains (Kp, Kd)
   - Increase height reward weight

2. **Training doesn't converge**:

   - Increase training timesteps
   - Adjust reward function weights
   - Check observation normalization

3. **Unstable behavior**:
   - Reduce control gains
   - Increase damping (Kd)
   - Add more stability rewards

### Debug Mode

Enable debug mode in the environment:

```python
env = SpiderRobotEnv(xml_file, debug=True)
```

This will print detailed information about:

- Initial robot state
- Joint limits
- Episode termination reasons
- Action statistics

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir=out/tensorboard/
```

### Videos

Check `out/videos/` for recorded episodes showing robot behavior.

### Logs

Training progress is printed to console and logged to TensorBoard.
