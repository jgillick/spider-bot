"""
Comprehensive Training Script for Spider Robot
Features:
- Parallel environments for faster training
- Curriculum learning with automatic progression
- Checkpointing and early stopping
- Learning rate scheduling
- Better evaluation and monitoring
- Video generation
"""

import os
import json
import traceback
import numpy as np
from moviepy import ImageSequenceClip
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder

from .environment import SpiderRobotEnv

DEFAULT_CONFIG = {
    "num_envs": 8,
    "total_timesteps": 10_000_000,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "n_steps": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.01,  # Reduced from 0.05 - simplified rewards need less exploration
    "max_grad_norm": 0.5,
    "network_arch": [
        256,
        256,
        128,
    ],
    "checkpoint_freq": 100_000,
    "eval_freq": 25_000,
    "stage_thresholds": [
        8000,
        12000,
        18000,
    ],  # Significantly reduced for simplified rewards
    "early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_min_improvement": 5.0,
    "generate_videos": True,
}


class CurriculumCallback(BaseCallback):
    """Callback for automatic curriculum progression."""

    def __init__(self, eval_env, stage_thresholds, verbose=0, out_dir=None):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.stage_thresholds = stage_thresholds
        self.current_stage = 1
        self.stage_episodes = 0
        self.stage_start_timestep = 0
        self.advance_requested = False  # Flag for advancing curriculum
        self.out_dir = out_dir
        self.xml_file = None  # Will be set later
        self.best_reward_per_stage = {
            1: -np.inf,
            2: -np.inf,
            3: -np.inf,
        }  # Track best rewards

    def request_advance(self):
        """Request curriculum advancement from evaluation callback."""
        self.advance_requested = True

    def get_current_stage(self):
        """Get current curriculum stage."""
        return self.current_stage

    def _on_step(self) -> bool:
        # Check if curriculum advancement was requested
        if self.advance_requested and self.current_stage < 3:
            # Record stage completion video
            if self.out_dir and self.xml_file:
                self._record_stage_video()

            self.current_stage += 1
            self.stage_start_timestep = self.num_timesteps
            self.advance_requested = False

            # Update all training environments
            if hasattr(self.training_env, "env_method"):
                try:
                    self.training_env.env_method(
                        "set_curriculum_stage", self.current_stage
                    )
                except:
                    pass  # Some vec envs don't support this

            print(
                f"\n🎯 Advanced to curriculum stage {self.current_stage} due to plateau at timestep {self.num_timesteps}"
            )

            # Log to tensorboard
            self.logger.record("curriculum/stage", self.current_stage)
            self.logger.record("curriculum/progression_timestep", self.num_timesteps)

        # Evaluate every 50k steps for natural progression check
        if self.n_calls % 50000 == 0 and self.n_calls > 0:
            # Manual evaluation to avoid evaluate_policy issues
            episode_rewards = []

            for _ in range(10):  # 10 episodes for curriculum evaluation
                obs = self.eval_env.reset()
                episode_reward = 0

                for _ in range(1000):  # Max 1000 steps per episode
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)

                    episode_reward += (
                        reward[0] if isinstance(reward, np.ndarray) else reward
                    )

                    if done[0] if isinstance(done, np.ndarray) else done:
                        break

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)

            # Update best reward for current stage
            if mean_reward > self.best_reward_per_stage[self.current_stage]:
                self.best_reward_per_stage[self.current_stage] = mean_reward

            # Check if ready to progress using BEST reward achieved
            if (
                self.best_reward_per_stage[self.current_stage]
                > self.stage_thresholds[self.current_stage - 1]
            ):
                if self.current_stage < 3:
                    self.current_stage += 1
                    self.stage_start_timestep = self.num_timesteps

                    # Update all training environments
                    # Note: Updating environments in SubprocVecEnv is complex
                    # For now, we'll track the stage and apply it on reset
                    try:
                        self.training_env.env_method(
                            "set_curriculum_stage", self.current_stage
                        )
                        self.eval_env.env_method(
                            "set_curriculum_stage", self.current_stage
                        )
                    except Exception as e:
                        print(
                            "⚠️ Error progressing training environments to the next stage!"
                        )
                        raise e

                    print(
                        f"\n🎯 Progressed to curriculum stage {self.current_stage} at timestep {self.num_timesteps}"
                    )
                    print(f"   Mean reward: {mean_reward:.2f}")

                    # Log to tensorboard
                    self.logger.record("curriculum/stage", self.current_stage)
                    self.logger.record(
                        "curriculum/progression_timestep", self.num_timesteps
                    )

                    # Note: The evaluation callback will detect this stage change and reset its metrics

            # Always log current performance
            self.logger.record("curriculum/mean_reward", mean_reward)
            self.logger.record("curriculum/current_stage", self.current_stage)

            # Add debugging to see why curriculum isn't advancing
            print(f"Current stage: {self.current_stage}")
            print(f"Current reward: {mean_reward}")
            print(
                f"Best reward for stage: {self.best_reward_per_stage[self.current_stage]}"
            )
            print(f"Stage threshold: {self.stage_thresholds[self.current_stage - 1]}")
            print(
                f"Should advance: {self.best_reward_per_stage[self.current_stage] > self.stage_thresholds[self.current_stage - 1]}"
            )

        return True

    def _record_stage_video(self):
        """Record a video showing the robot's performance at the end of a stage."""
        try:
            print(f"🎬 Recording stage {self.current_stage} completion video...")

            # Create video environment for this stage
            video_env = DummyVecEnv(
                [
                    lambda: SpiderRobotEnv(
                        self.xml_file,
                        render_mode="rgb_array",
                        camera_name="track",
                        width=1200,
                        height=800,
                        curriculum_stage=self.current_stage,
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
            stage_video_path = (
                f"{self.out_dir}/videos/stage_{self.current_stage}_completion"
            )
            video_env = VecVideoRecorder(
                video_env,
                video_folder=stage_video_path,
                record_video_trigger=lambda x: True,
                video_length=1000,  # 1000 steps for stage video
                name_prefix=f"stage_{self.current_stage}",
            )

            # Record one episode
            obs = video_env.reset()
            episode_reward = 0
            steps = 0

            for _ in range(1000):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = video_env.step(action)
                episode_reward += (
                    reward[0] if isinstance(reward, np.ndarray) else reward
                )
                steps += 1

                if done[0] if isinstance(done, np.ndarray) else done:
                    break

            video_env.close()
            print(f"✅ Stage {self.current_stage} video saved to {stage_video_path}")

        except Exception as e:
            print(f"⚠️ Failed to record stage {self.current_stage} video: {e}")

    def set_xml_file(self, xml_file):
        """Set the XML file path for video recording."""
        self.xml_file = xml_file


class CustomEvalCallback(BaseCallback):
    """Custom evaluation callback with curriculum advancement on plateau."""

    def __init__(
        self,
        eval_env,
        save_path,
        log_path,
        eval_freq,
        n_eval_episodes=10,
        patience=5,
        min_improvement=1.0,
        early_stopping=True,
        curriculum_callback=None,
        stage_thresholds=None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.save_path = save_path
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_eval_step = 0

        # Early stopping parameters
        self.patience = patience
        self.min_improvement = min_improvement
        self.early_stopping = early_stopping
        self.eval_history = []
        self.steps_without_improvement = 0
        self.best_step = 0

        # Curriculum integration
        self.curriculum_callback = curriculum_callback
        self.stage_thresholds = stage_thresholds or [15000.0, 25000.0, 35000.0]
        self.stage_best_rewards = {}  # Track best reward per stage

    def reset_for_new_stage(self):
        """Reset evaluation metrics for new curriculum stage."""
        current_stage = (
            self.curriculum_callback.current_stage if self.curriculum_callback else 1
        )
        print(f"   📊 Resetting metrics for stage {current_stage}")
        self.best_mean_reward = -np.inf
        self.steps_without_improvement = 0
        self.best_step = self.num_timesteps

    def _on_step(self) -> bool:
        # Check if curriculum stage has changed (from natural progression)
        if self.curriculum_callback and hasattr(self, "_last_known_stage"):
            current_stage = self.curriculum_callback.current_stage
            if current_stage != self._last_known_stage:
                print(
                    f"   🔄 Detected curriculum stage change: {self._last_known_stage} → {current_stage}"
                )
                self.reset_for_new_stage()
                self._last_known_stage = current_stage

                # CRITICAL: Update evaluation environment to new stage
                if hasattr(self.eval_env, "env_method"):
                    try:
                        self.eval_env.env_method("set_curriculum_stage", current_stage)
                        print(
                            f"   ✅ Updated evaluation environment to stage {current_stage}"
                        )
                    except Exception as e:
                        print(f"   ⚠️ Failed to update eval env stage: {e}")
        elif self.curriculum_callback:
            # Initialize tracking
            self._last_known_stage = self.curriculum_callback.current_stage

        # Check if we should evaluate based on total timesteps
        if self.num_timesteps >= self.last_eval_step + self.eval_freq:
            print(f"🎯 EVALUATION TRIGGERED at step {self.num_timesteps}")
            print(f"   Last eval: {self.last_eval_step}, freq: {self.eval_freq}")

            try:
                # Manual evaluation to avoid evaluate_policy issues
                print("   Running manual evaluation...")
                episode_rewards = []
                episode_lengths = []

                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    episode_reward = 0
                    episode_length = 0

                    for _ in range(2000):  # Max 2000 steps per episode
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

                # Get current stage
                current_stage = (
                    self.curriculum_callback.current_stage
                    if self.curriculum_callback
                    else 1
                )

                # Update curriculum callback's best reward tracking
                if (
                    self.curriculum_callback
                    and mean_reward
                    > self.curriculum_callback.best_reward_per_stage.get(
                        current_stage, -np.inf
                    )
                ):
                    self.curriculum_callback.best_reward_per_stage[current_stage] = (
                        mean_reward
                    )

                print(f"   📊 Evaluation Results (Stage {current_stage}):")
                print(f"      Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(
                    f"      Mean episode length: {mean_episode_length:.1f} ± {std_episode_length:.1f} steps"
                )
                print(f"      Best so far: {self.best_mean_reward:.2f}")

                # Check for improvement
                improvement = mean_reward - self.best_mean_reward
                if improvement > self.min_improvement:
                    print(f"   ✅ Improvement: +{improvement:.2f}")
                    self.best_mean_reward = mean_reward
                    self.steps_without_improvement = 0
                    self.best_step = self.num_timesteps

                    # Save best model
                    best_model_path = os.path.join(self.save_path, "best_model")
                    self.model.save(best_model_path)
                    print(f"   💾 Best model saved to {best_model_path}")

                else:
                    self.steps_without_improvement += 1
                    print(
                        f"   ⚠️ No improvement for {self.steps_without_improvement} evaluations"
                    )

                # Check for curriculum advancement or early stopping
                current_stage = (
                    self.curriculum_callback.current_stage
                    if self.curriculum_callback
                    else 1
                )

                if (
                    self.early_stopping
                    and self.steps_without_improvement >= self.patience
                ):
                    if current_stage < 3 and self.curriculum_callback:
                        # Advance curriculum instead of stopping
                        print(
                            f"   🎯 Plateau detected - advancing curriculum to stage {current_stage + 1}"
                        )
                        self.curriculum_callback.request_advance()
                        self.reset_for_new_stage()
                    elif current_stage >= 3:
                        # Final stage plateau - stop training
                        print(f"   🛑 Final stage plateau - stopping training")
                        return False

                # Log metrics
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.record("eval/mean_episode_length", mean_episode_length)
                self.logger.record("eval/std_episode_length", std_episode_length)
                self.logger.record("eval/best_mean_reward", self.best_mean_reward)
                self.logger.record(
                    "eval/steps_without_improvement", self.steps_without_improvement
                )
                self.logger.record("eval/curriculum_stage", current_stage)

                self.last_eval_step = self.num_timesteps

            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                traceback.print_exc()

        return True


class AdaptiveLearningRateCallback(BaseCallback):
    """Callback for adaptive learning rate scheduling."""

    def __init__(
        self, initial_lr=1e-4, min_lr=1e-6, decay_factor=0.95, patience=100000
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.last_improvement = 0
        self.current_lr = initial_lr

    def _on_step(self) -> bool:
        # Check every patience steps
        if self.n_calls % self.patience == 0 and self.n_calls > 0:
            # Get current learning rate
            if hasattr(self.model, "learning_rate"):
                current_lr = self.model.learning_rate
            else:
                current_lr = self.current_lr

            # Reduce learning rate
            new_lr = max(current_lr * self.decay_factor, self.min_lr)
            if new_lr != current_lr:
                self.model.learning_rate = new_lr
                self.current_lr = new_lr
                print(f"📉 Reduced learning rate to {new_lr:.2e}")

            # Log to tensorboard
            self.logger.record("train/learning_rate", new_lr)

        return True


def make_env(xml_file, out_dir, curriculum_stage=1, rank=0):
    """Create a single environment."""

    def _init():
        env = SpiderRobotEnv(xml_file, curriculum_stage=curriculum_stage)

        # Add TimeLimit wrapper to set max episode steps
        from gymnasium.wrappers import TimeLimit

        env = TimeLimit(env, max_episode_steps=2000)

        env = Monitor(env, f"{out_dir}/monitor_logs/env_{rank}")

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
                curriculum_stage=3,  # Show final stage performance
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
    video_env = VecVideoRecorder(
        video_env,
        video_folder=f"{out_dir}/videos/",
        record_video_trigger=lambda x: True,  # Record every episode
        video_length=2000,  # 2000 steps per video
        name_prefix="trained_spider",
    )

    print("🎥 Recording videos...")

    # Record multiple episodes
    for episode in range(3):
        print(f"   Recording episode {episode + 1}/3...")
        obs = video_env.reset()
        episode_reward = 0
        steps = 0

        for _ in range(2000):  # 2000 steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = video_env.step(action)
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            steps += 1

            if done[0] if isinstance(done, np.ndarray) else done:
                break

        print(f"   Episode {episode + 1}: {steps} steps, reward: {episode_reward:.2f}")

    video_env.close()
    print(f"✅ Videos saved to {out_dir}/videos/")

    # Also create a simple demonstration video with different camera angles
    create_demo_video(model, xml_file, out_dir)


def create_demo_video(model, xml_file, out_dir):
    """Create a demonstration video with different camera angles."""

    print("🎬 Creating demonstration video...")

    # Create environment for demo
    demo_env = SpiderRobotEnv(
        xml_file,
        render_mode="rgb_array",
        curriculum_stage=3,
        camera_name="track",
        width=1200,
        height=800,
    )

    # Run one episode and collect frames
    obs, _ = demo_env.reset()
    frames = []

    for _ in range(1000):  # 1000 steps for demo
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = demo_env.step(action)

        # Render frame
        frame = demo_env.render()
        if frame is not None:
            frames.append(frame)

        if done:
            break

    demo_env.close()

    # Save frames as video using moviepy
    if frames:
        try:
            # Create video from frames
            clip = ImageSequenceClip(frames, fps=20)
            demo_video_path = f"{out_dir}/videos/spider_demo.mp4"
            clip.write_videofile(demo_video_path, logger=None)
            print(f"✅ Demo video saved to {demo_video_path}")
        except Exception as e:
            print(f"⚠️ Error creating demo video: {e}")
            print("   Frames were captured but video creation failed")
    else:
        print("⚠️ No frames captured for demo video")


def train_spider(xml_file, config=None):
    """Train spider robot with improved infrastructure."""

    # Default configuration with curriculum advancement on plateau
    base_config = DEFAULT_CONFIG.copy()
    if config:
        base_config.update(config)
    config = base_config

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"./out/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{out_dir}/videos", exist_ok=True)
    os.makedirs(f"{out_dir}/monitor_logs", exist_ok=True)

    # Save configuration
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"📁 Output directory: {out_dir}")
    print(f"🔧 Configuration: {config['num_envs']} parallel environments")

    # Create parallel training environments
    # Use DummyVecEnv for stability (SubprocVecEnv can deadlock)
    if config["num_envs"] > 1:
        env = SubprocVecEnv(
            [
                make_env(xml_file, out_dir, curriculum_stage=1, rank=i)
                for i in range(config["num_envs"])
            ]
        )
    else:
        env = DummyVecEnv([make_env(xml_file, out_dir, curriculum_stage=1, rank=0)])

    # Add normalization
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )

    # Create evaluation environment with proper normalization
    eval_env = DummyVecEnv([make_env(xml_file, out_dir, curriculum_stage=1, rank=99)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards for evaluation
        clip_obs=10.0,
        training=False,
    )

    # Create model with PPO architecture
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        max_grad_norm=config["max_grad_norm"],
        policy_kwargs={"net_arch": config["network_arch"]},
        verbose=1,
        tensorboard_log=f"{out_dir}/tensorboard_logs/",
    )

    # Create callbacks
    curriculum_callback = CurriculumCallback(
        eval_env, config["stage_thresholds"], verbose=1, out_dir=out_dir
    )
    curriculum_callback.set_xml_file(xml_file)

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        save_path=f"{out_dir}/checkpoints",
        log_path=f"{out_dir}/tensorboard_logs/",
        eval_freq=config["eval_freq"],
        n_eval_episodes=10,
        patience=config["early_stopping_patience"],
        min_improvement=config["early_stopping_min_improvement"],
        early_stopping=config["early_stopping"],
        curriculum_callback=curriculum_callback,
        stage_thresholds=config["stage_thresholds"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=f"{out_dir}/checkpoints",
        name_prefix="spider_model",
    )

    lr_callback = AdaptiveLearningRateCallback(
        initial_lr=config["learning_rate"],
        patience=100000,
    )

    # Combine callbacks
    callbacks = CallbackList(
        [curriculum_callback, eval_callback, checkpoint_callback, lr_callback]
    )

    print("🎯 Starting training...")
    print(f"   Total timesteps: {config['total_timesteps']:,}")
    print(f"   Evaluation frequency: {config['eval_freq']:,}")
    print(f"   Checkpoint frequency: {config['checkpoint_freq']:,}")

    try:
        # Train the model
        model.learn(
            total_timesteps=config["total_timesteps"],
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
                action, _ = model.predict(obs, deterministic=True)
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
        if config.get("generate_videos", True):
            print("🎥 Generating training videos...")
            generate_training_videos(model, xml_file, out_dir, eval_env)

        # Save training summary
        summary = {
            "final_mean_reward": float(final_mean),
            "final_std_reward": float(final_std),
            "total_timesteps": config["total_timesteps"],
            "final_curriculum_stage": curriculum_callback.get_current_stage(),
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
    """Main function with multiple training options."""

    # Path to robot XML
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    xml_file = os.path.abspath(os.path.join(THIS_DIR, "../robot/SpiderBot.xml"))

    print("🕷️ Spider Robot Training")
    print("=" * 50)

    # Print configuration details
    print(f"\n🚀 Starting training")
    print(f"📁 Robot XML: {xml_file}")
    print(f"🔧 Configuration: {DEFAULT_CONFIG['num_envs']} parallel environments")
    print(f"⏱️ Training for {DEFAULT_CONFIG['total_timesteps']:,} timesteps")
    print()

    try:
        # Train the model
        model, output_dir = train_spider(xml_file, DEFAULT_CONFIG)
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
