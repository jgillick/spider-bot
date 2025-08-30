"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import os
import torch
from typing import Sequence, Any

import genesis as gs
from genesis.vis.camera import Camera
from genesis.utils.geom import transform_by_quat, inv_quat

from genesis_forge import GenesisEnv, EnvMode
from genesis_forge.managers import (
    CommandManager,
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
    DofPositionActionManager,
    ContactManager,
)
from genesis_forge.utils import (
    robot_projected_gravity, 
    robot_ang_vel, 
    robot_lin_vel, 
    links_idx_by_name_pattern,
)
from genesis_forge.mdp import rewards, terminations


INITIAL_BODY_POSITION = [0.0, 0.0, 0.14]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_XML = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))


class SpiderRobotEnv(GenesisEnv):
    """
    SpiderBot environment for Genesis
    """

    camera: Camera = None

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int | None = 6,
        headless: bool = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
            headless=headless,
        )

        self._curriculum_phase = 1

        # Cache position buffers
        self.base_init_pos = torch.tensor(INITIAL_BODY_POSITION, device=gs.device)
        self.base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device).reshape(
            1, -1
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )

        """
        Configuration
        """
        # Define the DOF actuators
        self.action_manager = DofPositionActionManager(
            self,
            joint_names=".*",
            default_pos={
                # Hip joints
                "Leg[1-2]_Hip": -1.0,
                "Leg[3-6]_Hip": 1.0,
                "Leg[7-8]_Hip": -1.0,
                # Femur joints
                "Leg[1-8]_Femur": 0.5,
                # Tibia joints
                "Leg[1-8]_Tibia": 0.6,
            },
            pd_kp=50,
            pd_kv=0.5,
            max_force=8.0,
            # stiffness=0.1,
            # frictionloss=0.1,
            # noise_scale=0.02,
        )

        # Command manager: instruct the robot to move in a certain direction
        self.height_command = CommandManager(
            self,
            range={
                "height": [0.12, 0.15],
            },
        )
        self.velocity_command = VelocityCommandManager(
            self,
            # Starting ranges should be small, while robot is learning to stand
            range={
                "lin_vel_x": [-0.5, 0.5],
                "lin_vel_y": [-0.5, 0.5],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.02,
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        # Contact managers: Legs should not come in contact with anything
        self.bad_touch_contact = ContactManager(
            self,
            link_names=[
                "Leg[1-8]_Tibia_BadTouch",
            ],
        )
        self.foot_contact_manager = ContactManager(
            self,
            link_names=[
                "Leg[1-8]_Tibia_Foot"
           ],
        )

        # Rewards
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "Linear Z velocity": {
                    "weight": -10.0,
                    "fn": rewards.lin_vel_z,
                },
                "Base height": {
                    "weight": -1000.0,
                    "fn": rewards.base_height,
                    "params": {
                        # "target_height": 0.135,
                        "height_command": self.height_command,
                    },
                },
                "Similar to default": {
                    "weight": -1.0,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "dof_action_manager": self.action_manager,
                    },
                },
                "Action rate": {
                    "weight": -0.05,
                    "fn": rewards.action_rate,
                },
                "Cmd linear velocity": {
                    "weight": 20.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Cmd angular velocity": {
                    "weight": 10.0,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                    },
                },
                "Flat orientation": {
                    "weight": -50.0,
                    "fn": rewards.flat_orientation_l2,
                },
                "Bad touch": {
                    "weight": 0.0, #-10.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.bad_touch_contact,
                    },
                },
                "Foot contact (some)": {
                    "weight": 0.0, #5.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "min_contacts": 4,
                    },
                },
                "Foot contact (all)": {
                    "weight": 0.0, #0.1,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "min_contacts": 8,
                    },
                },
                "Leg angle": {
                    "weight": -1.5,
                    "fn": self._penalize_leg_angle,
                },
            },
        )

        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "Timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "Bad angle": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 0.7,
                    },
                },
            },
        )

    """
    Properties
    """

    @property
    def action_space(self):
        return self.action_manager.action_space

    """
    Operations
    """

    def construct_scene(self) -> gs.Scene:
        """Add the robot to the scene."""
        scene = super().construct_scene(
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_self_collision=True,
            )
        )

        # add robot
        self.robot = scene.add_entity(
            gs.morphs.MJCF(
                file=SPIDER_XML,
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Add camera
        self.camera = scene.add_camera(
            pos=(-2.5, -1.5, 1.0), res=(1280, 960), fov=40, debug=True
        )

        return scene

    def build(self) -> None:
        """
        Builds the scene after all entities have been added (via construct_scene).
        This operation is required before running the simulation.
        """
        super().build()

        # Track robot with camera
        self.camera.follow_entity(self.robot, fixed_axis=(None, None, 1.0))
        self.camera.set_pose(lookat=self.robot.get_pos())

        # Fetch foot links
        self._foot_links_idx = links_idx_by_name_pattern(self.robot, "Leg[1-8]_Tibia_Foot")
        self._foot_link_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device)
        self._foot_link_gravity = self._foot_link_gravity.unsqueeze(0).expand(self.num_envs, len(self._foot_links_idx), 3)


    def observations(self) -> torch.Tensor:
        """Generate a list of observations for each environment."""

        # If this is being called before the first step, actions should all be zero
        actions = self.actions
        if actions is None:
            actions = torch.zeros((self.num_envs, self.action_manager.num_actions), device=gs.device)

        self.obs_buf = torch.cat(
            [
                self.height_command.command, # 1
                self.velocity_command.command,  # 3
                robot_ang_vel(self),  # 3
                robot_lin_vel(self),  # 3
                robot_projected_gravity(self),  # 3
                self.action_manager.get_dofs_position(noise=0.01),  # 24
                self.action_manager.get_dofs_velocity(noise=0.1),  # 24
                actions,  # 24
                # self.action_manager.get_dofs_force(noise=0.01, clip_to_max_force=True),  # 24
            ],
            dim=-1,
        )
        return self.obs_buf

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        super().step(actions)
        info = {"logs": {}}

        # Execute the actions and a simulation step
        self.action_manager.step(actions)
        self.scene.step()

        # Calculate contact forces
        self.bad_touch_contact.step()
        self.foot_contact_manager.step()

        #  Keep the camera looking at the robot
        self.camera.set_pose(lookat=self.robot.get_pos())

        # Termination, rewards
        terminated, truncated, reset_env_idx = self.termination_manager.step()
        reward = self.reward_manager.step()

        # Command manager
        self.velocity_command.step()
        self.height_command.step()

        # Update curriculum
        # self._update_curriculum(info)

        # Log metrics
        info["logs"]["Metrics / Leg Contact"] = rewards.has_contact(self, self.bad_touch_contact)
        info["logs"]["Metrics / Foot Contact"] = rewards.has_contact(self, self.foot_contact_manager)
        info["logs"]["Metrics / Curriculum level"] = self._curriculum_phase

        # Finish up
        self.reset(reset_env_idx)
        self.observations()
        return self.obs_buf, reward, terminated, truncated, info

    def reset(
        self,
        envs_idx: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or more environments.
        """
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        if envs_idx.numel() > 0:
            # Managers
            self.height_command.reset(envs_idx)
            self.velocity_command.reset(envs_idx)
            self.termination_manager.reset(envs_idx)
            self.reward_manager.reset(envs_idx)
            self.action_manager.reset(envs_idx)
            self.bad_touch_contact.reset(envs_idx)
            self.foot_contact_manager.reset(envs_idx)

            # Reset robot
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_quat[envs_idx] = self.base_init_quat
            self.robot.zero_all_dofs_velocity(envs_idx)
            self.robot.set_pos(self.base_pos[envs_idx], envs_idx=envs_idx)
            self.robot.set_quat(self.base_quat[envs_idx], envs_idx=envs_idx)

        obs = self.observations()
        return obs, {}

    def render(self):
        pass

    def close(self):
        """Close the environment."""
        self.scene.reset()

    """
    Implementation
    """

    def _update_curriculum(self, info: dict[str, Any]):
        """
        Update the training curriculum based on the current step or environment performance.
        """
        # We've (hopefully) learned to stand, let's try to walk
        if self.step_count == 15_000:
            self._curriculum_phase += 1
            self.velocity_command.range["lin_vel_x"] = [-1.5, 1.5]
            self.velocity_command.range["lin_vel_y"] = [-1.5, 1.5]
            self.velocity_command.range["ang_vel_z"] = [-1.5, 1.5]
            self.set_max_episode_length(round(self.max_episode_length_sec * 1.5))

            self.reward_manager.cfg["Similar to default"]["weight"] = 0.5
            # self.reward_manager.cfg["Bad touch"]["weight"] = -10.0
            self.reward_manager.cfg["Foot contact (some)"]["weight"] = 5
            self.reward_manager.cfg["Foot contact (all)"]["weight"] = 0.1
    
    def _penalize_leg_angle(self, _env: GenesisEnv):
        """Penalize the tibia bending in too far and under the robot."""
        target_angle = 0.0

        quats = self.robot.get_links_quat(links_idx_local=self._foot_links_idx)

        # Transform link frames to world gravity frames
        inv_quats = inv_quat(quats)
        gravity_in_links = transform_by_quat(self._foot_link_gravity, inv_quats)
        uprightness = -gravity_in_links[..., 2]

        # Add up all the uprightness values less than zero
        return torch.abs(torch.sum(uprightness * (uprightness < target_angle), dim=1))
