#!/bin/bash
#
# A simple wrapper to run scripts in the Isaac environment.
#
# Usage:
#   ./isaac.sh --help
#

SCRIPT_DIR="$(dirname "$(realpath $0)")"
source $SCRIPT_DIR/env.cfg

$ISAACLAB_ROOT/isaaclab.sh $@