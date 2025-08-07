#!/bin/bash
# Install the spider-bot robot tasks package in the IsaacLab environment.

# Load environment
source env.cfg

# Install the package
$ISAACLAB_ROOT/isaaclab.sh -p -m pip install -e ./extension
