# Spider Robot Training

This directory contains the training code for the spider robot, separated into modular components for better organization.

## File Structure

- **`spider_env.py`** - The main environment class (`SpiderRobotEnv`) and helper functions
- **`train.py`** - Training script using PPO algorithm
- **`test.py`** - Testing and evaluation script for trained models
- **`deploy.py`** - Deployment script for real hardware (placeholder)
- **`env.py`** - Original combined file (kept for reference)

## Quick Start

### 1. Training

```bash
# Train the spider robot
python train.py
```

This will:

- Create output directories (`out/`, `out/videos/`, `out/tensorboard/`)
- Train for 500,000 timesteps by default
- Save the model to `out/spider_robot_final`
- Record videos every 25,000 steps
- Log training progress to TensorBoard

### 2. Testing

```bash
# Test a trained model
python test.py
```

This will:

- Load the trained model
- Evaluate performance over 5 episodes
- Run a few episodes with rendering
- Print detailed statistics

### 3. Hardware Deployment

```bash
# Deploy to real hardware (placeholder)
python deploy.py
```

**Note**: This is a placeholder implementation. You'll need to:

- Implement actual ODrive controller connections
- Add sensor reading code
- Implement safety systems
- Test thoroughly in simulation first

## Environment Details

### Action Space

- **24 joint positions** (8 legs × 3 joints each)
- Uses PD control with Kp=10.0, Kd=0.5
- Torque limits: ±8.0 N⋅m

### Observation Space (87 dimensions)

- Joint positions (31)
- Joint velocities (30)
- IMU data (10): orientation, angular velocity, linear acceleration
- Body position/orientation (7)
- Contact forces (8)
- Gait phase (1)

### Reward Function

- Forward progress (moderate weight)
- Height maintenance (critical)
- Upright orientation (critical)
- Body stability
- Gait coordination
- Energy efficiency
- Survival bonus

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
