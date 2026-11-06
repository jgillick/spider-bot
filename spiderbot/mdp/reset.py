import torch
import genesis as gs
from dataclasses import dataclass

from genesis_forge import GenesisEnv
from genesis_forge.managers import ResetMdpFn, TerrainManager
from genesis.engine.entities import RigidEntity
from genesis.utils.geom import transform_quat_by_quat


@dataclass(kw_only=True, eq=False)
class random_ground_pose(ResetMdpFn):
    """
    Reset the robot's joints, height and tilt to a random pre-generated resting pose.

    The poses are recorded on flat ground, so the pose height is applied relative to the
    terrain height under the robot. The pose rotation is applied on top of the robot's
    current rotation, so this should run after a reset function that sets the
    position/heading of the robot (i.e. `reset.randomize_terrain_position`).

    Args:
        poses: List of poses, each a dict with "pos", "quat" and "joints" (name -> position).
        terrain_manager: The terrain manager, used to get the terrain height under the robot.
        probability: The probability that a resetting environment gets a random pose.
                     The rest are left as they are (i.e. the default pose).
        height_margin: Extra height added to the pose to prevent the feet starting
                       inside uneven terrain.
    """

    poses: list[dict]
    terrain_manager: TerrainManager
    probability: float = 1.0
    height_margin: float = 0.005

    def build(self):
        if len(self.poses) == 0:
            raise RuntimeError(
                "No poses found. Generate them with: python -m spiderbot.locomotion.generate_random_ground_pos"
            )

        dof_names = list(self.poses[0]["joints"].keys())
        joints = {joint.name: joint for joint in self.entity.joints}
        self._dofs_idx = [joints[name].dofs_idx_local[0] for name in dof_names]

        self._height = torch.tensor(
            [pose["pos"][2] for pose in self.poses], device=gs.device, dtype=gs.tc_float
        )
        self._quat = torch.tensor(
            [pose["quat"] for pose in self.poses], device=gs.device, dtype=gs.tc_float
        )
        self._dof_pos = torch.tensor(
            [[pose["joints"][name] for name in dof_names] for pose in self.poses],
            device=gs.device,
            dtype=gs.tc_float,
        )

    def __call__(self, env: GenesisEnv, entity: RigidEntity, envs_idx: torch.Tensor):
        envs_idx = envs_idx[torch.rand(len(envs_idx), device=gs.device) < self.probability]
        if len(envs_idx) == 0:
            return

        idx = torch.randint(0, self._quat.shape[0], (len(envs_idx),), device=gs.device)

        # Height relative to the terrain
        pos = entity.get_pos(envs_idx).clone()
        terrain_height = self.terrain_manager.get_terrain_height(pos[:, 0], pos[:, 1])
        pos[:, 2] = terrain_height + self._height[idx] + self.height_margin
        entity.set_pos(pos, envs_idx=envs_idx, zero_velocity=True)

        # Apply the pose tilt on top of the current heading
        quat = transform_quat_by_quat(self._quat[idx], entity.get_quat(envs_idx))
        entity.set_quat(quat, envs_idx=envs_idx, zero_velocity=True)

        entity.set_dofs_position(
            position=self._dof_pos[idx],
            dofs_idx_local=self._dofs_idx,
            envs_idx=envs_idx,
        )
        entity.zero_all_dofs_velocity(envs_idx)
