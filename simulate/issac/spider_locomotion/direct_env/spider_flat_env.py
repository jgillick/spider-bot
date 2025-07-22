from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from .spider_flat_cfg import SpiderFlatEnvCfg

class SpiderLocomotionFlatDirectEnv(DirectRLEnv):
    cfg: SpiderFlatEnvCfg

    def __init__(
        self,
        cfg: SpiderFlatEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.set_debug_vis(self.cfg.debug_vis)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "bad_touch",
                "tipped_over",
                "bottom_contact",
            ]
        }
        self._episode_metrics = {
            "robot_height": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("Body")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_Tibia_Foot")
        self._bottom_body_ids, _ = self._contact_sensor.find_bodies(".*_Hip_Bracket")
        self._bad_touch_bodies, _ = self._contact_sensor.find_bodies(
            ".*_BadTouch"
        )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Scale the actions from from -1 to 1, to the full joint ranges
        self._actions = actions.clone().clamp(-1.0, 1.0)
        target_positions = math_utils.unscale_transform(
            actions,
            self._robot.data.soft_joint_pos_limits[:, :, 0],
            self._robot.data.soft_joint_pos_limits[:, :, 1],
        )
        self._processed_actions = (target_positions)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # yaw rate tracking
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self._feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self._bottom_body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        rewards = {
            "track_lin_vel_xy_exp": self._mdp_commanded_direction_reward,
            "track_ang_vel_z_exp": yaw_rate_error_mapped
            * self.cfg.yaw_rate_reward_scale
            * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error
            * self.cfg.ang_vel_reward_scale
            * self.step_dt,
            "dof_torques_l2": joint_torques
            * self.cfg.joint_torque_reward_scale
            * self.step_dt,
            "dof_acc_l2": joint_accel
            * self.cfg.joint_accel_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            "feet_air_time": air_time
            * self.cfg.feet_air_time_reward_scale
            * self.step_dt,
            "undesired_contacts": contacts
            * self.cfg.undesired_contact_reward_scale
            * self.step_dt,
            "flat_orientation_l2": flat_orientation
            * self.cfg.flat_orientation_reward_scale
            * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_metrics["robot_height"] += self._robot.data.root_pos_w[:, 2]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Is the robot putting force on "BadTouch" bodies
        bad_touch = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self._bad_touch_bodies], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

        # Is the robot putting force on the bottom body
        # bottom_contact = torch.any(
        #     torch.max(
        #         torch.norm(net_contact_forces[:, :, self._bottom_body_ids], dim=-1),
        #         dim=1,
        #     )[0]
        #     > 1.0,
        #     dim=1,
        # )

        # The robot has tipped over
        tipped_over = (
            torch.acos(-self._robot.data.projected_gravity_b[:, 2]).abs() > 0.7
        )

        died = torch.any(torch.stack([bad_touch, tipped_over]), dim=0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -self.cfg.max_command_velocity, self.cfg.max_command_velocity
        )

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        for key in self._episode_metrics.keys():
            episodic_sum_avg = torch.mean(self._episode_metrics[key][env_ids])
            extras[f"Episode_Metrics/{key}"] = episodic_sum_avg / self.max_episode_length
            self._episode_metrics[key][env_ids] = 0.0
        extras["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace( 
                    prim_path="/Visuals/Command/velocity_current"
                ))
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(BLUE_ARROW_X_MARKER_CFG.replace( 
                    prim_path="/Visuals/Command/velocity_current"
                ))
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    
    def _mdp_commanded_velocity_reward(self) -> torch.Tensor:
        """
        Reward for moving in the commanded direction and at the commanded velocity.
        """
        lin_vel_error = torch.sum(
            torch.square(
                self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]
            ),
            dim=1,
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        return lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt

    def _mdp_commanded_direction_reward(self) -> torch.Tensor:
        """
        Reward for moving in the commanded direction.
        The reward is proportional to how fast the robot is moving in that direction.
        This is not based on the magnitude of the commanded velocity (unlike _mdp_commanded_velocity_reward)
        """
        cmd_norm = torch.norm(self._commands[:, :2], dim=1, keepdim=True) + 1e-6
        cmd_dir = self._commands[:, :2] / cmd_norm
        speed_in_cmd_dir = torch.sum(self._robot.data.root_lin_vel_b[:, :2] * cmd_dir, dim=1)

        return speed_in_cmd_dir * self.cfg.lin_vel_reward_scale * self.step_dt
