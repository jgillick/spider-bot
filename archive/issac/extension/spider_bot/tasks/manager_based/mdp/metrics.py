import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

def robot_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Return the height of the robot's base"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2]

def height_diff_from_target(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Return the absolute height difference from the target height"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 2] - target_height)

def max_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Return the maximum force on contact sensors across all selected sensors"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Get max force for each sensor
    max_force_per_sensor = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    # Get max force across all sensors
    max_force = torch.max(max_force_per_sensor, dim=1)[0]
    return max_force

def mean_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Return the mean force on contact sensors across all selected sensors"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # Get mean force for each sensor
    mean_force_per_sensor = contact_forces.norm(dim=-1).mean(dim=1)
    # Get mean force across all sensors
    mean_force = torch.mean(mean_force_per_sensor, dim=1)
    return mean_force
