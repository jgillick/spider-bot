#!/bin/bash
#
# Run the tensorboard on an isaac RL training session.
#

source ./env.cfg

$ISAACLAB_ROOT/isaaclab.sh -p -m tensorboard.main --logdir=logs/rsl_rl/spider_bot/
