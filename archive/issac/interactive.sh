#!/bin/bash
#
# Run the interactive environment defined at scripts/interactive_env.py
# in the IsaacLab environment.
#

# Load environment
source env.cfg

$ISAACLAB_ROOT/isaaclab.sh -p scripts/interactive_env.py $@
