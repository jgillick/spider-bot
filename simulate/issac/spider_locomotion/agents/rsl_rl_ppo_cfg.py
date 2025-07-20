from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpiderBotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # Increased from 24 for more DOF
    max_iterations = 3000  # Increased from 1500 for more complex learning
    save_interval = 50
    experiment_name = "spider_bot"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.5,  # Increased from 1.0 for better exploration with more DOF
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Increased from 0.005 for better exploration
        num_learning_epochs=6,  # Increased from 5 for more complex policy
        num_mini_batches=6,  # Increased from 4 for better gradient estimates
        learning_rate=8.0e-4,  # Slightly reduced from 1.0e-3 for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SpiderBotFlatPPORunnerCfg(SpiderBotRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000  # Increased from 300 for more DOF
        self.experiment_name = "spider_bot_flat"
        # Increased network capacity for 24 DOF robot
        self.policy.actor_hidden_dims = [
            256,
            256,
            128,
        ]  # Increased from [128, 128, 128]
        self.policy.critic_hidden_dims = [
            256,
            256,
            128,
        ]  # Increased from [128, 128, 128]
        # Reduced initial noise for flat terrain but still higher than original
        self.policy.init_noise_std = 1.2  # Increased from 1.0 for better exploration
