#!/bin/bash
#
# A simple wrapper to run scripts in the Isaac environment.
#
# Usage:
#   ./isaac.sh --help
#

source env.cfg

$ISAACLAB_ROOT/isaaclab.sh $@