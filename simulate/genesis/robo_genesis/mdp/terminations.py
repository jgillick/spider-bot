import torch
from robo_genesis.genesis_env import GenesisEnv
from robo_genesis.utils import robot_projected_gravity


def timeout(env: GenesisEnv):
    """
    Terminate the environment if the episode length exceeds the maximum episode length.
    """
    return env.episode_length > env.max_episode_length


def bad_orientation(
    env: GenesisEnv,
    limit_angle: float = 0.7,
) -> torch.Tensor:
    """
    Terminate the environment if the robot is tipping over too much.

    This function uses projected gravity to detect when the robot has tilted
    beyond a safe threshold. When the robot is perfectly upright, projected
    gravity should be [0, 0, -1] in the body frame. As the robot tilts,
    the x,y components increase, indicating roll and pitch angles.

    Args:
        env: The Genesis environment containing the robot
        limit_angle: Maximum allowed tilt angle in radians (default: 0.7 ~ 40 degrees)

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    # Get the projected gravity vector in body frame
    projected_gravity = robot_projected_gravity(env)
    projected_gravity_xy = projected_gravity[:, :2]

    # Calculate the magnitude of tilt (distance from perfectly upright)
    # This directly corresponds to the tilt angle
    tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)

    # Convert tilt magnitude to angle (tilt_magnitude = sin(tilt_angle))
    # For small angles: sin(angle) ≈ angle, so we can use the magnitude directly
    # For larger angles, we can use asin(tilt_magnitude) for more accuracy
    tilt_angle = torch.asin(
        torch.clamp(tilt_magnitude, max=0.99)
    )  # Clamp to avoid asin(1)

    # Terminate if tilt angle exceeds the limit
    return tilt_angle > limit_angle


def root_height_below_minimum(
    env: GenesisEnv,
    minimum_height: float = 0.05,
) -> torch.Tensor:
    """
    Terminate the environment if the robot's base height falls below a minimum threshold.

    Args:
        env: The Genesis environment containing the robot
        minimum_height: Minimum allowed base height in meters

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    base_pos = env.robot.get_pos()
    return base_pos[:, 2] < minimum_height
