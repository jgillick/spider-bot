#!/bin/bash
# Run a training session locally.
#
#
#######################################################
# CONFIGURATION
#######################################################
HEADLESS=1
NUM_ENVS=40
MAX_ITERATIONS=500

# Video settings (only when HEADLESS=1)
VIDEO_LENGTH=2000
VIDEO_INTERVAL=500

TRAINING_TASK="SpiderBot-Flat-v0"
TRAINING_SCRIPT="./scripts/rsl_rl/train.py"

#######################################################
# MAIN
#######################################################

source env.cfg

# Cleanup on exit
trap "exit" INT TERM
trap "kill 0" EXIT

export HYDRA_FULL_ERROR=1

if [ $HEADLESS -eq 1 ]; then
  $ISAACLAB_ROOT/isaaclab.sh -p $TRAINING_SCRIPT \
    --task $TRAINING_TASK \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --headless \
    --verbose \
    --enable_cameras \
    --video \
    --video_length $VIDEO_LENGTH \
    --video_interval $VIDEO_INTERVAL
else
  $ISAACLAB_ROOT/isaaclab.sh -p $TRAINING_SCRIPT \
    --task $TRAINING_TASK \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS
fi
