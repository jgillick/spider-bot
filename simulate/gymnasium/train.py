"""
Comprehensive Training Script for Spider Robot
"""

import os
import json
import shutil
import argparse
import traceback
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CallbackList,
    BaseCallback,
)
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from QuietVideoRecorder import QuietVideoRecorder
from environment import SpiderRobotEnv


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

XML_FILE = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))
OUT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "./logs/"))

DEFAULT_NUM_ENVS = 8

CONFIG = {
    "total_timesteps": 10_000_000,
    "max_episode_steps": 1_000,
    "eval_episodes": 10,
    "evaluation_steps": 500,
    "eval_video_steps_length": 500,
    "eval_freq": 50_000,
    "video_freq": 100_000,
    "generate_videos": True,
    "plateau_after_n_evals": None,
    "learning_decay": {
        "initial_lr": 3e-4,
        "final_lr": 1e-4,
        "decay_steps": 3_000_000,
    },
    "PPO_params": {
        "verbose": 1,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": [
                128,
                256,
                64,
            ]
        },
    },
    "SAC_params": {
        "verbose": 1,
        "learning_rate": 3e-4,
        "buffer_size": 2_000_000,  # Increased from 1M
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.01,  # Increased from 0.005 for faster target updates
        "gamma": 0.99,
        "ent_coef": 0.1,  # Fixed entropy coefficient
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": True,  # Use State Dependent Exploration
        "sde_sample_freq": 64,  # How often to sample new noise
        "policy_kwargs": {
            "net_arch": [256, 256, 128],
            "log_std_init": -2.0,  # Start with smaller exploration
        },
    },
}


class EvalCallback(BaseCallback):
    """Callback to evaluate the model at regular intervals."""

    def __init__(
        self,
        eval_env,
        checkpoint_path,
        video_path,
        verbose=0,
        out_dir=None,
        train_env=None,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_env = train_env
        self.checkpoint_path = checkpoint_path
        self.video_path = video_path
        self.next_eval_step = CONFIG["eval_freq"]
        self.next_video_step = CONFIG["video_freq"]

        self.best_mean_reward = -np.inf
        self.steps_without_improvement = 0
        self.best_step = None
        self.last_mean = 0

    def evaluate_model(self):
        """Manually evaluate the model at this point."""
        # Sync normalization statistics before evaluation
        if self.train_env is not None and hasattr(self.train_env, "obs_rms"):
            if hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms = self.train_env.obs_rms
        if self.train_env is not None and hasattr(self.train_env, "ret_rms"):
            if hasattr(self.eval_env, "ret_rms"):
                self.eval_env.ret_rms = self.train_env.ret_rms

        episode_rewards = []
        episode_lengths = []
        for _ in range(CONFIG["eval_episodes"]):
            obs = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0

            for _ in range(CONFIG["evaluation_steps"]):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)

                episode_reward += (
                    reward[0] if isinstance(reward, np.ndarray) else reward
                )
                episode_length += 1

                if done[0] if isinstance(done, np.ndarray) else done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = np.std(episode_lengths)

        return (mean_reward, std_reward, mean_episode_length, std_episode_length)

    def _on_step(self) -> bool:
        is_best = False
        if self.num_timesteps >= self.next_eval_step:
            print("-" * 50)
            print(f"Step: {self.num_timesteps:,}")
            print("📊 Running evaluation...")
            self.next_eval_step += CONFIG["eval_freq"]
            mean_reward, std_reward, mean_episode_length, std_episode_length = (
                self.evaluate_model()
            )
            mean_reward = float(mean_reward)
            best_improvement = mean_reward - self.best_mean_reward
            last_eval_improvement = mean_reward - self.last_mean
            is_best = best_improvement > 0
            self.last_mean = mean_reward
            diff_sign = "+" if last_eval_improvement > 0 else ""

            print(f"   Mean reward: {mean_reward:,.2f} ± {std_reward:,.2f}")
            print(f"   Dif to last: {diff_sign}{last_eval_improvement:.2f}")
            print(
                f"   Mean length: {mean_episode_length:.1f} ± {std_episode_length:.1f} steps"
            )

            print(f"✅ Best so far: {self.best_mean_reward:,.2f}")
            if not is_best and self.best_step is not None:
                self.steps_without_improvement += 1
                print(
                    f"⚠️  No improvement for {self.steps_without_improvement} evaluation(s)"
                )

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_episode_length", mean_episode_length)
            self.logger.record("eval/std_episode_length", std_episode_length)

            # Check for improvement
            if is_best:
                print(f"🚀 Overall Improvement: +{best_improvement:,.2f}")
                self.best_mean_reward = mean_reward
                self.steps_without_improvement = 0
                self.best_step = self.num_timesteps

                # Save best model
                best_model_path = os.path.join(self.checkpoint_path, "best_model")
                self.model.save(best_model_path)
            # Has plateaued
            elif (
                CONFIG["plateau_after_n_evals"] is not None
                and self.steps_without_improvement >= CONFIG["plateau_after_n_evals"]
            ):
                print(f"🛑 Final stage plateau - stopping training")
                return False

        # Record video
        is_video_timestep = self.num_timesteps >= self.next_video_step
        if is_video_timestep or is_best:
            if is_video_timestep:
                self.next_video_step += CONFIG["video_freq"]
            video_path = self._record_eval_video()
            if is_best and video_path is not None:
                shutil.copyfile(video_path, f"{self.video_path}/best.mp4")

        return True

    def _record_eval_video(self):
        """Record a video showing the robot's performance at the end of a stage."""
        try:
            print(f"🎬 Recording video...")

            # Create video environment for this stage
            video_env = DummyVecEnv(
                [
                    lambda: SpiderRobotEnv(
                        XML_FILE,
                        render_mode="rgb_array",
                        camera_name="track",
                        width=1200,
                        height=800,
                    )
                ]
            )

            # Apply normalization if available
            if hasattr(self.eval_env, "obs_rms"):
                video_env = VecNormalize(
                    video_env,
                    norm_obs=True,
                    norm_reward=False,
                    clip_obs=10.0,
                    training=False,
                )
                video_env.obs_rms = self.eval_env.obs_rms

            # Wrap with video recorder
            video_env = QuietVideoRecorder(
                video_env,
                video_folder=self.video_path,
                record_video_trigger=lambda x: True,
                video_length=CONFIG["eval_video_steps_length"],
                name_prefix=f"eval",
            )
            video_env.step_id = self.num_timesteps

            # Record one episode
            obs = video_env.reset()
            steps = 0

            for _ in range(CONFIG["eval_video_steps_length"]):
                action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
                obs, reward, done, info = video_env.step(action)
                steps += 1
                if done[0] if isinstance(done, np.ndarray) else done:
                    break

            video_env.close()
            return video_env.video_path
        except Exception as e:
            print(f"⚠️ Failed to record video: {e}")

        return None


class LearningRateScheduler(BaseCallback):
    """Callback to schedule learning rate decay during training."""

    def __init__(
        self, initial_lr=1e-3, final_lr=1e-4, decay_steps=5_000_000, verbose=0
    ):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps
        self.current_lr = initial_lr

    def _on_step(self) -> bool:
        if self.num_timesteps <= self.decay_steps:
            progress = self.num_timesteps / self.decay_steps
            self.current_lr = (
                self.initial_lr + (self.final_lr - self.initial_lr) * progress
            )

            # Update learning rate based on algorithm type
            if hasattr(self.model.policy, "optimizer"):
                # PPO has a single optimizer
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = self.current_lr
            elif hasattr(self.model, "actor") and hasattr(self.model, "critic"):
                # SAC has separate actor and critic optimizers
                actor = getattr(self.model, "actor")
                critic = getattr(self.model, "critic")
                for param_group in actor.optimizer.param_groups:
                    param_group["lr"] = self.current_lr
                for param_group in critic.optimizer.param_groups:
                    param_group["lr"] = self.current_lr

            # Logging
            if self.verbose and self.num_timesteps % 100_000 == 0:
                print(f"📉 Learning rate: {self.current_lr:.2e}")

        return True


def make_env(out_dir, thread, video=False):
    """Create a single environment."""

    def _init():
        env = SpiderRobotEnv(
            XML_FILE,
            render_mode="rgb_array" if video else None,
        )
        env = TimeLimit(env, max_episode_steps=CONFIG["max_episode_steps"])
        env = Monitor(env, f"{out_dir}/monitor_logs/env_{thread}")
        return env

    return _init


def generate_training_videos(model, xml_file, out_dir, eval_env=None):
    """Generate videos of the trained model performing."""

    # Create video environment
    video_env = DummyVecEnv(
        [
            lambda: SpiderRobotEnv(
                xml_file,
                render_mode="rgb_array",
                camera_name="track",
                width=1200,
                height=800,
            )
        ]
    )

    # Apply same normalization as training
    if eval_env is not None:
        # Copy normalization stats from evaluation environment
        video_env = VecNormalize(
            video_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
        )
        # Try to copy stats if available
        if hasattr(eval_env, "obs_rms"):
            video_env.obs_rms = eval_env.obs_rms
        if hasattr(eval_env, "ret_rms"):
            video_env.ret_rms = eval_env.ret_rms

    # Wrap with video recorder
    video_env = QuietVideoRecorder(
        video_env,
        video_folder=f"{out_dir}/videos/",
        record_video_trigger=lambda x: True,
        video_length=CONFIG["max_episode_steps"],
        name_prefix="trained_spider",
    )

    print("🎥 Recording videos...")

    # Record multiple episodes
    for episode in range(3):
        print(f"  Recording episode {episode + 1}/3...")
        obs = video_env.reset()
        episode_reward = 0
        steps = 0

        for _ in range(CONFIG["max_episode_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = video_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward

            steps += 1
            if done[0] if isinstance(done, np.ndarray) else done:
                break

        print(f"  Episode {episode + 1}: {steps} steps, reward: {episode_reward:.2f}")

    video_env.close()
    print(f"✅ Videos saved to {out_dir}/videos/")


def train_spider(algorithm="PPO", num_envs=DEFAULT_NUM_ENVS):
    """Train spider robot with improved infrastructure."""

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{OUT_ROOT}/{timestamp}_{algorithm}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{out_dir}/videos", exist_ok=True)
    os.makedirs(f"{out_dir}/monitor_logs", exist_ok=True)
    os.makedirs(f"{out_dir}/code", exist_ok=True)

    print()
    print(f"📜 Algorithm: {algorithm}")
    print(f"📁 Output directory: {out_dir}")
    print(f"🏋️‍♀️ Training for {CONFIG['total_timesteps']:,} timesteps")
    print(f"👯 Parallel environments: {num_envs}")
    print(f"⏰ Evaluation frequency: {CONFIG['eval_freq']:,}")
    print(f"🎥 Video frequency: {CONFIG['video_freq']:,}")
    print()

    # Save configuration
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Save current training code
    shutil.copyfile(f"{THIS_DIR}/train.py", f"{out_dir}/code/train.py")
    shutil.copyfile(f"{THIS_DIR}/environment.py", f"{out_dir}/code/environment.py")

    # Create parallel training environments
    # Use DummyVecEnv for stability (SubprocVecEnv can deadlock)
    if num_envs > 1:
        env = SubprocVecEnv([make_env(out_dir, thread=i) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_env(out_dir, thread=0)])

    # Consistent normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)
    eval_env = DummyVecEnv([make_env(out_dir, thread="env")])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=True, training=False, clip_reward=10.0
    )

    # Create model
    tensorboard_log = f"{out_dir}/tensorboard/"
    if algorithm == "PPO":
        learning_rate = CONFIG["PPO_params"]["learning_rate"]
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=tensorboard_log,
            **CONFIG["PPO_params"],
        )
    elif algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log=tensorboard_log,
            **CONFIG["SAC_params"],
        )

    # Create callbacks
    callback_list = []
    checkpoint_path = f"{out_dir}/checkpoints"
    video_path = f"{out_dir}/videos"
    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        checkpoint_path=checkpoint_path,
        video_path=video_path,
        train_env=env,
    )
    callback_list.append(eval_callback)

    # Add learning rate scheduler to address training slowdown
    lr_scheduler = LearningRateScheduler(
        verbose=1,
        initial_lr=CONFIG["learning_decay"]["initial_lr"],
        final_lr=CONFIG["learning_decay"]["final_lr"],
        decay_steps=CONFIG["learning_decay"]["decay_steps"],
    )
    callback_list.append(lr_scheduler)

    # Combine callbacks
    callbacks = CallbackList(callback_list)

    try:
        # Train the model
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=callbacks,
            progress_bar=True,
        )

        print("✅ Training completed successfully!")

        # Final evaluation
        print("🎯 Running final evaluation...")
        final_rewards = []
        for episode in range(10):
            obs = eval_env.reset()
            episode_reward = 0
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)  # type: ignore[arg-type]
                obs, reward, done, info = eval_env.step(action)
                episode_reward += (
                    reward[0] if isinstance(reward, np.ndarray) else reward
                )
                if done[0] if isinstance(done, np.ndarray) else done:
                    break
            final_rewards.append(episode_reward)

        final_mean = np.mean(final_rewards)
        final_std = np.std(final_rewards)
        print(f"📊 Final Performance: {final_mean:.2f} ± {final_std:.2f}")

        # Save final model
        final_model_path = os.path.join(out_dir, "final_model")
        model.save(final_model_path)
        print(f"💾 Final model saved to {final_model_path}")

        # Generate videos if requested
        if CONFIG.get("generate_videos", True):
            print("🎥 Generating training videos...")
            generate_training_videos(model, XML_FILE, out_dir, eval_env)

        # Save training summary
        summary = {
            "final_mean_reward": float(final_mean),
            "final_std_reward": float(final_std),
            "total_timesteps": CONFIG["total_timesteps"],
        }
        with open(f"{out_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📁 All results saved to: {out_dir}")
        print(
            "📊 View training progress with: tensorboard --logdir",
            f"{out_dir}/tensorboard_logs/",
        )

        return model, out_dir

    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        raise

    finally:
        # Clean up
        env.close()
        eval_env.close()


def main():
    """Main CLI entry point for the training script."""

    parser = argparse.ArgumentParser(description="Train SpiderBot to walk")
    parser.add_argument("algorithm", help="PPO or SAC", default="PPO")
    parser.add_argument(
        "-n",
        "--num-envs",
        help="Number of parallel environments to run",
        default=DEFAULT_NUM_ENVS,
        type=int,
    )
    args = parser.parse_args()

    print("🕷️ Spider Robot Training")
    print("=" * 50)

    try:
        # Train the model
        model, output_dir = train_spider(
            algorithm=args.algorithm, num_envs=args.num_envs
        )
        print(f"✅ Training completed successfully!")
        print(f"📁 Results saved to: {output_dir}")

        # Additional analysis
        print("\n📊 Training Summary:")
        print("   Check TensorBoard for detailed metrics")
        print("   Review videos to see robot behavior")
        print("   Use test.py to analyze the trained model")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
