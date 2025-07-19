#!/bin/bash
#
# Run a headless training session locally.
#
# Assumptions:
#   - Isaac lab & sim are installed and available in the active python environment
#   - The spider_locomotion directory is symlinked to `<ISAACLAB_ROOT>source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion`
#
#

#######################################################
# CONFIGURATION
#######################################################
NUM_ENVS=512
VIDEO_LENGTH=1000
VIDEO_INTERVAL=100

TRAINING_TASK="Isaac-SpiderLocomotion-Flat-v0"
TRAINING_SCRIPT="./scripts/reinforcement_learning/rsl_rl/train.py"

ISAACLAB_ROOT="$HOME/Documents/Projects/isaac/IsaacLab"

#######################################################
# MAIN
#######################################################

cd $ISAACLAB_ROOT

# Cleanup on exit
trap "exit" INT TERM
trap "kill 0" EXIT

# Start the tensorboard server
start_tensorboard() {
  mkdir -p logs
  ./isaaclab.sh -p -m tensorboard.main --logdir=logs --bind_all
}
start_tensorboard &

./isaaclab.sh -p $TRAINING_SCRIPT \
  --task $TRAINING_TASK \
  --num_envs $NUM_ENVS \
  --headless \
  --verbose \
  --enable_cameras \
  --video \
  --video_length $VIDEO_LENGTH \
  --video_interval $VIDEO_INTERVAL
