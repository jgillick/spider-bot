#!/bin/bash
# Script to add the mjcf to the headless kit
#
# Usage: ./patch_headless_kit.sh <kit file>
# Example: ./patch_headless_kit.sh ~/IsaacLab/apps/isaaclab.python.headless.kit
#
set -e

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path to patch_headless_kit.sh>"
    exit 1
fi

FILE="$1"
PYTHON_DEPENDENCY="isaacsim.asset.importer.mjcf"

# Check if the file exists
if [[ ! -f "$FILE" ]]; then
    echo "Error: File '$FILE' not found!"
    exit 1
fi

# Check if the string already exists in the file
if ! grep -q "$PYTHON_DEPENDENCY" "$FILE"; then
    echo "Adding '$PYTHON_DEPENDENCY' to $FILE"

    # Check if [dependencies] section exists
    if grep -q "^\[dependencies\]" "$FILE"; then
        # Create a backup of the original file
        cp "$FILE" "$FILE.backup"

        # Use sed to insert the line after [dependencies]
        sed -i "/^\[dependencies\]/a \"$PYTHON_DEPENDENCY\" = {}" "$FILE"
    else
        echo "Error: [dependencies] section not found in $FILE"
        echo "Please ensure the file has a [dependencies] section."
        exit 1
    fi
fi
