from genesis_forge.managers import (
    PositionActionManager,
    RewardManager,
    TerminationManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    CommandManager,
)
from genesis_forge.managers.actuator import ActuatorManager, NoisyValue
from genesis_forge.mdp import reset, rewards, terminations, observations

from spiderbot.environment import BaseSpiderRobotEnv, Terrain, EnvMode
from spiderbot.mdp.foot_angle import FootAngleMdp


class SpiderRobotHeightEnv(BaseSpiderRobotEnv):
    """
    A spider training environment for height control
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_sec: int | None = 6,
        headless: bool = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_sec,
            terrain="flat",
        )

    """
    Operations
    """

    def config(self):
        """
        Configure the environment managers.
        """
        # Foot angle monitor
        self.foot_angle_mdp = FootAngleMdp(self, foot_name_pattern="[RL][1-4]_Foot")

        ##
        # Robot manager
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.randomize_terrain_position,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                        "height_offset": 0.118,
                    },
                },
            },
        )

        ##
        # Actuators and Actions
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=".*",
            default_pos={
                "R1_Hip": 0.9,
                "R2_Hip": 0.2,
                "R3_Hip": -0.2,
                "R4_Hip": -0.9,
                "L1_Hip": -0.9,
                "L2_Hip": -0.2,
                "L3_Hip": 0.2,
                "L4_Hip": 0.0,
                "[RL][1-4]_Femur": -0.4,
                "[RL][1-4]_Tibia": 0.5,
            },
            kp=NoisyValue(30, 5),
            kv=NoisyValue(1.5, 0.5),
            max_force=NoisyValue(10.0, 1.0),
            frictionloss=NoisyValue(0.1, 0.05),
            damping=NoisyValue(0.0381, 0.0001),
            armature=0.0020,
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
        # Contact managers

        # Foot/step contact manager
        self.foot_contact_manager = ContactManager(
            self,
            link_names=["[RL][1-4]_Foot"],
            with_entity_attr="terrain",
            track_air_time=True,
        )

        # Detect self contacts
        self.self_contact = ContactManager(
            self,
            entity_attr="robot",
            link_names=[
                "[RL][1-4]_Femur",
                "[RL][1-4]_Tibia",
                "[RL][1-4]_Foot",
                ".*_Motor",
            ],
            with_entity_attr="robot",
            with_links_names=[
                "[RL][1-4]_Femur",
                "[RL][1-4]_Tibia",
                "[RL][1-4]_Foot",
                ".*_Motor",
            ],
        )

        ##
        # Command manager
        self.height_command_manager = CommandManager(
            self, range=(0.09, 0.15), resample_time_sec=2.0
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(
            self,
            cfg={
                "height": {
                    "weight": -40.0,
                    "fn": rewards.base_height,
                    "params": {
                        "height_command": self.height_command_manager,
                    },
                },
                "similar_to_default": {
                    "weight": -0.001,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "action_manager": self.action_manager,
                    },
                },
                "flat_orientation": {
                    "weight": -1.5,
                    "fn": rewards.flat_orientation_l2,
                },
                "ang_vel_xy_l2": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2,
                },
                "leg_angle": {
                    "weight": -0.02,
                    "fn": self.foot_angle_mdp.reward,
                },
                "stable_footing": {
                    "weight": 0.02,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                "self_contact": {
                    "weight": -0.05,
                    "fn": rewards.contact_force,
                    "params": {
                        "contact_manager": self.self_contact,
                    },
                },
                "action_rate": {
                    "weight": -0.04,
                    "fn": rewards.action_rate_l2,
                },
                "action_acceleration": {
                    "weight": -1e-04,
                    "fn": rewards.action_acceleration_l2,
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
                    "fn": terminations.timeout,
                },
                "self_contact": {
                    "fn": terminations.contact_force,
                    "params": {
                        "contact_manager": self.self_contact,
                        "threshold": 2.0,
                    },
                },
                "foot_angle": {
                    "fn": self.foot_angle_mdp.terminate,
                    "params": {
                        "angle_threshold": -0.75,
                    },
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            history_len=2,
            cfg={
                "command": {
                    "fn": self.height_command_manager.observation,
                },
                "imu_lin_acc_ang_vel": {
                    "fn": self.imu_observation,
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
                    "fn": lambda env: self.action_manager.raw_actions,
                },
            },
        )
        ObservationManager(
            self,
            history_len=2,
            name="privileged",
            cfg={
                "height_sensor": {
                    "fn": self.height_sensor_observation,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "foot_contacts": {
                    "fn": observations.contact_force,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                "control_force": {
                    "fn": lambda env: self.robot.get_dofs_control_force(),
                    "scale": 0.1,
                },
                "force": {
                    "fn": lambda env: self.robot.get_dofs_force(),
                    "scale": 0.1,
                },
            },
        )

    def build(self):
        super().build()
        self.foot_angle_mdp.build()
