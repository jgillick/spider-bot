from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SpiderBotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Target training batch size of ~98,304 (98,304 / num parallel envs = num_steps_per_env)
    # Based on: https://ar5iv.labs.arxiv.org/html/2109.11978
    num_steps_per_env = 164  
    max_iterations = 2000 
    save_interval = 50
    experiment_name = "spider_bot"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=2.0,  # Increased from 1.5 for more exploration
        # Larger network for 24 DOF spider robot (vs 12 DOF ANYmal)
        # Increased capacity to handle more complex coordination patterns
        actor_hidden_dims=[512, 256, 256, 128],
        critic_hidden_dims=[512, 256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.05,  # Increased from 0.02 to break out of local optimum
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=6.0e-4,  # Reduced from 1.0e-3 for more stable learning
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
        self.max_iterations = 3000
        self.experiment_name = "spider_bot_flat"
        self.policy.actor_hidden_dims = [
            256,
            256,
            128,
        ]
        self.policy.critic_hidden_dims = [
            256,
            256,
            128,
        ]
