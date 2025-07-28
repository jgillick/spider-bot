#!/bin/bash
#
# Run a headless training session locally.
#
# Assumptions:
#   - Isaac lab & sim are installed and available in the active python environment
#   - The spider_locomotion directory is symlinked to `<ISAACLAB_ROOT>source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/spider_locomotion`
#
#

SCRIPT_DIR="$(dirname "$(realpath $0)")"
source $SCRIPT_DIR/env.cfg

#######################################################
# CONFIGURATION
#######################################################
HEADLESS=1
NUM_ENVS=600
MAX_ITERATIONS=500

# Video settings (only when HEADLESS=1)
VIDEO_LENGTH=1000
VIDEO_INTERVAL=500

# TRAINING_TASK="Isaac-SpiderLocomotion-Flat-v0"
TRAINING_TASK="Isaac-SpiderLocomotion-Flat-v0"
TRAINING_SCRIPT="./scripts/reinforcement_learning/rsl_rl/train.py"

#######################################################
# MAIN
#######################################################

cd $ISAACLAB_ROOT

# Cleanup on exit
trap "exit" INT TERM
trap "kill 0" EXIT

export HYDRA_FULL_ERROR=1

if [ $HEADLESS -eq 1 ]; then
  ./isaaclab.sh -p $TRAINING_SCRIPT \
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
  ./isaaclab.sh -p ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-SpiderLocomotion-Direct-v0 --num_envs $NUM_ENVS --max_iterations $MAX_ITERATIONS 
fi
