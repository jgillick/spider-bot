import torch

from genesis_forge import GenesisEnv
from genesis_forge.managers import (
    PositionActionManager,
    RewardManager,
    TerminationManager,
    CommandManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    VelocityCommandManager,
    ActuatorMonitor,
    MotorSpec,
    PowerMonitor,
)
from genesis_forge.managers.actuator import ActuatorManager, NoisyValue
from genesis_forge.managers.command.velocity_command import VelocityCommandRange
from typing import cast
from genesis_forge.mdp import reset, rewards, terminations, observations


from spiderbot.environment import BaseSpiderRobotEnv, Terrain, EnvMode
from spiderbot.mdp.foot_angle import FootAngleMdp
from spiderbot.mdp.friction import randomize_terrain_friction
from spiderbot.mdp.reset import random_ground_pose
from spiderbot.locomotion.ground_positions import RANDOM_GROUND_POSES

CURRICULUM_CHECK_EVERY_STEPS = 1200
CURRICULUM_AVG_SAMPLES = 5
CURRICULUM_MAX_LEVEL = 50
CURRICULUM_MIN_EPISODE_LENGTH_RATIO = 0.6
CURRICULUM_COOLDOWN_STEPS = 1200

# Fraction of resets that start from a random ground pose instead of the default pose
RANDOM_POSE_PROBABILITY_START = 0.2
RANDOM_POSE_PROBABILITY_INC = 0.05
RANDOM_POSE_PROBABILITY_MAX = 0.8

# Energy penalty, in reward per watt of estimated motor power (see spiderbot/motor/power.py).
# It starts small and grows with the curriculum, so the policy learns to walk before it
# learns to save power (too strong too early and lying still wins).
ENERGY_WEIGHT_START = -2e-05
ENERGY_WEIGHT_INC = -1e-05
ENERGY_WEIGHT_MAX = -5e-04

# Base height range. It starts around the default pose height (0.118)
# and widens with the curriculum, toward the effective standing range.
HEIGHT_RANGE_START = (0.11, 0.13)
HEIGHT_RANGE_INC = (-0.005, 0.005)
HEIGHT_RANGE_LIMIT = (0.10, 0.17)

# Joint groups for the power and actuator logs
JOINT_GROUPS = {
    "hip": "[RL][1-4]_Hip",
    "femur": "[RL][1-4]_Femur",
    "tibia": "[RL][1-4]_Tibia",
}

# GIM8108-8 leg motors (see spiderbot/motor/power.py)
TORQUE_CONSTANT =  1.0  # N·m/A, output side
PHASE_RESISTANCE = 0.439  # ohm
LEG_MOTOR = MotorSpec(
    torque_constant=1.0,
    phase_resistance=0.439,
)
SUPPLY_VOLTAGE = 44.4  # V, 12S nominal


class SpiderRobotEnv(BaseSpiderRobotEnv):
    """
    SpiderBot locomotion environment
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_sec: int | None = 15,
        headless: bool = True,
        mode: EnvMode = "train",
        terrain: Terrain = "flat",
        height_sensor: bool = False,
    ):
        self.curriculum_level = 1
        self.curriculum_samples = []
        self.curriculum_ep_length_samples = []
        self.next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS

        self.max_velocity_x = 1.0
        self.max_velocity_y = 1.0
        self.max_velocity_z = 1.0
        self.velocity_inc = 0.025
        if terrain == "flat":
            self.max_velocity_x = 1.2
            self.max_velocity_y = 1.0
            self.max_velocity_z = 1.0
            self.velocity_inc = 0.025

        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_sec,
            headless=headless,
            mode=mode,
            terrain=terrain,
            height_sensor=height_sensor,
        )

    """
    Operations
    """

    def config(self):
        """
        Configure the environment managers.
        """
        super().config()

        # Foot angle monitor
        self.foot_angle_mdp = FootAngleMdp(self, foot_name_pattern="[RL][1-4]_Foot")

        ##
        # Robot manager
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "position": {
                    "fn": (
                        reset.randomize_terrain_position(
                            terrain_manager=self.terrain_manager,
                            height_offset=0.118,
                            subterrain=self.curriculum_terrain,
                        )
                        if self.mode == "train"
                        else reset.position(position=(0.0, 0.0, 0.118))
                    ),
                },
                # Must come after "position", since it builds on the position and heading.
                # The probability is increased by the curriculum.
                "random_ground_pose": {
                    "fn": random_ground_pose(
                        poses=RANDOM_GROUND_POSES,
                        terrain_manager=self.terrain_manager,
                        probability=RANDOM_POSE_PROBABILITY_START,
                    ),
                },
                # Setting the mass is slow (~120ms per call), so only re-randomize it
                # every ~10 iterations, instead of on every reset.
                "random_mass": {
                    "fn": reset.randomize_link_mass_shift(
                        mass_range=(0.0, 5.0),
                        link_name="Body",
                        min_step_count_between_reset=250,
                    ),
                },
                "friction": {
                    "fn": randomize_terrain_friction(
                        terrain=self.terrain,
                        link_names=["[RL][1-4]_Foot"],
                        friction_range=(0.4, 1.5),
                    ),
                },
            },
        )

        ##
        # Actuators and Actions
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=".*",
            default_pos={
                "R1_Hip": NoisyValue(0.9, 0.1),
                "R2_Hip": NoisyValue(0.2, 0.1),
                "R3_Hip": NoisyValue(-0.2, 0.1),
                "R4_Hip": NoisyValue(-0.9, 0.1),
                "L1_Hip": NoisyValue(-0.9, 0.1),
                "L2_Hip": NoisyValue(-0.2, 0.1),
                "L3_Hip": NoisyValue(0.2, 0.1),
                "L4_Hip": NoisyValue(0.9, 0.1),
                "[RL][1-4]_Femur": NoisyValue(-0.4, 0.1),
                "[RL][1-4]_Tibia": NoisyValue(0.5, 0.1),
            },
            kp=NoisyValue(30, 5),
            kv=NoisyValue(2.0, 0.5),
            max_force=NoisyValue(18.0, 1.0),
            armature=0.0020,
            frictionloss=NoisyValue(0.1, 0.05),
            damping=NoisyValue(0.0381, 0.0001),
        )
        self.action_manager = PositionActionManager(
            self,
            delay_step=1,
            scale=0.25,
            use_default_offset=True,
            soft_limit_scale_factor=0.95,
            actuator_manager=self.actuator_manager,
        )

        ##
        # Monitors
        self.power_monitor = PowerMonitor(
            self,
            actuator_manager=self.actuator_manager,
            motors=LEG_MOTOR,
            supply_voltage=SUPPLY_VOLTAGE,
            dof_groups=JOINT_GROUPS,
        )
        self.actuator_monitor = ActuatorMonitor(
            self,
            actuator_manager=self.actuator_manager,
            dof_groups=JOINT_GROUPS,
            saturation_threshold=0.95,
        )

        ##
        # Contact managers

        # Foot/step contact manager
        self.foot_contact_manager = ContactManager(
            self,
            link_names=["[RL][1-4]_Foot"],
            with_entity=self.terrain,
            track_air_time=True,
        )

        # Detect self contacts
        self.self_contact = ContactManager(
            self,
            entity=self.robot,
            link_names=[
                "[RL][1-4]_Femur",
                "[RL][1-4]_Tibia",
                "[RL][1-4]_Foot",
                ".*_Motor",
            ],
            with_entity=self.robot,
            with_links_names=[
                "[RL][1-4]_Femur",
                "[RL][1-4]_Tibia",
                "[RL][1-4]_Foot",
                ".*_Motor",
            ],
        )

        ##
        # Command manager
        self.vel_command_manager = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-0.6, 0.6),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-1.0, 1.0),
            },
            resample_time_sec=4.0,
            standing_probability=0.15,
            debug_visualizer=True,
            debug_visualizer_cfg={"envs_idx": [0], "ang_arc_offset": 0.15},
        )
        self.height_command_manager = CommandManager(
            self,
            range={"height": HEIGHT_RANGE_START},
            resample_time_sec=4.0,
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(  # type: ignore[arg-type]
            self,
            cfg={
                "cmd_linear_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.vel_command_manager,
                    ),
                },
                "cmd_angular_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.vel_command_manager,
                    ),
                },
                "height": {
                    "weight": 0.5,
                    "fn": rewards.base_height_exp(
                        height_command=self.height_command_manager,
                        # Reward falls to 1/e at 3cm of error
                        sensitivity=0.03**2,
                        terrain_manager=self.terrain_manager,
                    ),
                },
                # "similar_to_default": {
                #     "weight": -0.01,
                #     "fn": rewards.dof_similar_to_default(
                #         actuator_manager=self.actuator_manager,
                #     ),
                # },
                "flat_orientation": {
                    "weight": -1.5,
                    "fn": rewards.flat_orientation_l2(),
                },
                "ang_vel_xy_l2": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2(),
                },
                "foot_air_time": {
                    "weight": 0.5,
                    "fn": rewards.feet_air_time(
                        contact_manager=self.foot_contact_manager,
                        vel_cmd_manager=self.vel_command_manager,
                        time_threshold=0.2,
                        time_threshold_max=0.8,
                    ),
                },
                "feet_ground_time": {
                    "weight": -1.0,
                    "fn": rewards.feet_ground_time(
                        contact_manager=self.foot_contact_manager,
                        time_threshold=0.3,
                    ),
                },
                "feet_max_air_time": {
                    "weight": -1.0,
                    "fn": self.feet_max_air_time_reward,
                    "params": {
                        "max_air_time": 1.0,
                    },
                },
                # Stand still when the velocity command is stopped
                "stand_still": {
                    "weight": -1.6e-3,
                    "fn": rewards.dof_velocity_l2(
                        actuator_manager=self.actuator_manager,
                    ),
                    "when": self.vel_command_manager.when_stopped()
                },
                "feet_slide": {
                    "weight": -0.1,
                    "fn": rewards.feet_slide(
                        contact_manager=self.foot_contact_manager,
                    ),
                },
                # "leg_angle": {
                #     "weight": -0.02,
                #     "fn": self.foot_angle_mdp.reward,
                # },
                # "self_contact": {
                #     "weight": -0.05,
                #     "fn": rewards.contact_force(
                #         contact_manager=self.self_contact,
                #     ),
                # },
                "action_rate": {
                    "weight": -0.02,
                    "fn": rewards.action_rate_l2(),
                },
                "action_acceleration": {
                    "weight": -4e-03,
                    "fn": rewards.action_acceleration_l2(),
                },
                "energy": {
                    "weight": ENERGY_WEIGHT_START,
                    "fn": rewards.electrical_power(power_monitor=self.power_monitor),
                },
            },
        )

        ##
        # Terminations
        self.termination_manager = TerminationManager(
            self,
            term_cfg={
                "timeout": {
                    "time_out": True,
                    "fn": terminations.timeout(),
                },
                "out_of_bounds": {
                    "time_out": True,
                    "fn": terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager,
                    ),
                },
                "bad_orientation": {
                    "fn": terminations.bad_orientation(
                        limit_angle=70,
                    ),
                },
                "self_contact": {
                    "fn": terminations.contact_force(
                        contact_manager=self.self_contact,
                        threshold=2.0,
                    ),
                },
                "foot_angle": {
                    "fn": self.foot_angle_mdp.terminate,
                    "params": {
                        "angle_threshold": -0.75,
                    },
                },
                "dof_velocity_limit": {
                    "fn": terminations.dof_velocity_limit(
                        unit="rpm",
                        threshold=250.0,
                        actuator_manager=self.actuator_manager,
                    ),
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            history_len=5,
            cfg={
                "command": {
                    "fn": self.vel_command_manager.observation,
                },
                "height_command": {
                    "fn": self.height_command_manager.observation,
                },
                "imu_lin_acc_ang_vel": {
                    "fn": self.imu_observation,
                },
                "imu_projected_gravity": {
                    "fn": observations.imu_projected_gravity(
                        imu_sensor=self.imu,
                    ),
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                    "noise": 0.01,
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.1,
                    "noise": 0.1,
                },
                "actions": {
                    "fn": observations.current_actions(),
                },
            },
        )
        ObservationManager(
            self,
            history_len=5,
            name="privileged",
            cfg={
                "height_sensor": {
                    "fn": self.height_sensor_observation,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "foot_contacts": {
                    "fn": observations.contact_force(
                        contact_manager=self.foot_contact_manager,
                    ),
                },
                "control_force": {
                    "fn": lambda env: self.actuator_manager.get_dofs_control_force(
                        clip_to_max_force=True
                    ),
                    "scale": 0.1,
                },
                "force": {
                    "fn": lambda env: self.robot.get_dofs_force(),
                    "scale": 0.1,
                },
                "power": {
                    "fn": observations.dof_power(
                        power_monitor=self.power_monitor
                    ),
                    "scale": 0.01,
                }
            },
        )

    def build(self):
        super().build()
        self.foot_angle_mdp.build()

    def step(self, actions: torch.Tensor):
        """
        Perform a step in the environment.
        """
        obs, reward, terminated, truncated, extras = super().step(actions)

        # Logging
        extras = self.log_curriculum(extras)
        extras = self.log_metrics(extras)

        # Finish up
        return obs, reward, terminated, truncated, extras

    def reset(self, env_ids: list[int] | None = None):
        if env_ids is not None:
            self._last_reset_mean_ep_length = (
                self.episode_length[env_ids].float().mean().item()
            )
        result = super().reset(env_ids)
        if env_ids is not None:
            self.update_curriculum()
        return result

    def log_curriculum(self, extras: dict):
        extras["episode"]["Curriculum / curriculum_level"] = self.curriculum_level
        vel_range = cast(VelocityCommandRange, self.vel_command_manager.range)
        extras["episode"]["Curriculum / max_velocity_x"] = vel_range["lin_vel_x"][1]
        extras["episode"]["Curriculum / max_velocity_y"] = vel_range["lin_vel_y"][1]
        height_range = cast(dict, self.height_command_manager.range)["height"]
        extras["episode"]["Curriculum / min_height"] = height_range[0]
        extras["episode"]["Curriculum / max_height"] = height_range[1]
        extras["episode"]["Curriculum / energy_weight"] = self.reward_manager[
            "energy"
        ].weight
        extras["episode"]["Curriculum / random_pose_probability"] = (
            self.robot_manager.on_reset["random_ground_pose"].fn.probability
        )

        return extras

    def log_metrics(self, extras: dict):
        base_pos = self.robot_manager.base_pos
        height_offset = self.terrain_manager.get_terrain_height(
            base_pos[:, 0], base_pos[:, 1]
        )
        extras["episode"]["Metrics / height_avg"] = torch.mean(
            base_pos[:, 2] - height_offset
        )

        return extras

    def feet_max_air_time_reward(
        self,
        env: GenesisEnv,
        max_air_time: float,
    ) -> torch.Tensor:
        """Penalise any foot that stays airborne longer than ``max_air_time``.

        Unlike ``feet_air_time`` (which fires once at touchdown), this is a
        continuous per-step penalty that grows linearly with every additional
        second the foot remains in the air beyond the threshold.  This gives
        the policy an ongoing gradient to bring a retracted leg back down,
        which ``feet_air_time`` alone cannot provide.

        Should be registered with a negative weight.  Set ``max_air_time``
        comfortably above the expected swing duration (e.g. 1.5× the
        ``time_threshold_max`` used in ``feet_air_time``) so that normal
        stepping is unaffected.

        Args:
            env: The Genesis Forge environment.
            max_air_time: Seconds a foot may be airborne before the penalty
                          starts accumulating.

        Returns:
            Per-environment sum of excess air time across all feet,
            shape ``(n_envs,)``.
        """
        excess = (self.foot_contact_manager.current_air_time - max_air_time).clamp(
            min=0.0
        )
        return excess.sum(dim=-1)

    def update_curriculum(self):
        """
        Check the curriculum
        """
        if self.curriculum_level >= CURRICULUM_MAX_LEVEL:
            return
        # Limit how often we check/update the curriculum
        if self.step_count < self.next_curriculum_check_step:
            return

        # Collect samples over multiple resets to prevent momentary spikes causing a level up
        if len(self.curriculum_samples) < CURRICULUM_AVG_SAMPLES:
            cmd_linear_vel = self.reward_manager.last_episode_mean_reward(
                "cmd_linear_vel", before_weight=True
            )
            self.curriculum_samples.append(cmd_linear_vel)
            self.curriculum_ep_length_samples.append(self._last_reset_mean_ep_length)
            if len(self.curriculum_samples) < CURRICULUM_AVG_SAMPLES:
                return

        cmd_linear_avg = sum(self.curriculum_samples) / len(self.curriculum_samples)
        ep_length_avg = sum(self.curriculum_ep_length_samples) / len(
            self.curriculum_ep_length_samples
        )
        min_ep_length = (
            self.max_episode_length_steps * CURRICULUM_MIN_EPISODE_LENGTH_RATIO
        )

        # Level up when the policy both tracks commands well and survives long enough
        if cmd_linear_avg > 0.8 and ep_length_avg >= min_ep_length:
            self.curriculum_level += 1
            self.vel_command_manager.increment_range(
                "lin_vel_x", self.velocity_inc, limit=self.max_velocity_x
            )
            self.vel_command_manager.increment_range(
                "lin_vel_y", self.velocity_inc, limit=self.max_velocity_y
            )
            self.vel_command_manager.increment_range(
                "ang_vel_z", self.velocity_inc, limit=self.max_velocity_z
            )

            # Widen the height range
            self.height_command_manager.increment_range(
                "height", HEIGHT_RANGE_INC, limit=HEIGHT_RANGE_LIMIT
            )

            # Start more episodes from a random pose
            random_pose = self.robot_manager.on_reset["random_ground_pose"].fn
            random_pose.probability = min(
                random_pose.probability + RANDOM_POSE_PROBABILITY_INC,
                RANDOM_POSE_PROBABILITY_MAX,
            )

            # Ask for more energy efficiency
            self.reward_manager["energy"].increment_weight(
                ENERGY_WEIGHT_INC, limit=ENERGY_WEIGHT_MAX
            )

            # Increase the gait reward
            # self.reward_manager["gait_diagonal"].increment_weight(0.05, limit=0.30)
            # self.reward_manager["gait_lateral"].increment_weight(0.05, limit=0.2)

            # Cooldown: wait longer before the next level-up can be considered
            self.next_curriculum_check_step = (
                self.step_count + CURRICULUM_COOLDOWN_STEPS
            )
        else:
            self.next_curriculum_check_step = (
                self.step_count + CURRICULUM_CHECK_EVERY_STEPS
            )

        self.curriculum_samples = []
        self.curriculum_ep_length_samples = []

    def curriculum_terrain(self) -> str | None:
        """
        Select the terrain type for the environment.
        """
        if self.terrain_type == "mixed" and self.curriculum_level <= 3:
            return "flat"
        return None
