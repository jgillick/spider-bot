#!/bin/bash
#
# Run the tensorboard on an isaac RL training session.
#
#

source ./env.cfg

cd $ISAACLAB_ROOT
mkdir -p logs
./isaaclab.sh -p -m tensorboard.main --logdir=logs --bind_all
