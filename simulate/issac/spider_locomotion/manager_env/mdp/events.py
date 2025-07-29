import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedEnv

def reset_joints( env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        print("RESETTING JOINTS")
        asset = env.scene[asset_cfg.name]
        joint_pos = asset.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids, asset_cfg.joint_ids].clone()
        print(joint_pos)
        asset.write_joint_state_to_sim(
            joint_pos.view(len(env_ids), -1),
            joint_vel.view(len(env_ids), -1),
            env_ids=env_ids,
            joint_ids=asset_cfg.joint_ids,
        )