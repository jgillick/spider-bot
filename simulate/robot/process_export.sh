#!/bin/bash
set -e

EXPORT_DIR="./export"

echo "Processing XML..."

# Mujoco
python update_mujoco.py --ground --light --head --imu SpiderBot.xml

# URDF
python update_urdf.py SpiderBot.urdf

###
# Move mesh files, if they exist
#
echo "Updating URDF meshes..."
if [ -d "./meshes/urdf" ]; then
  rm -r ./meshes/urdf
fi
mkdir -p "./meshes/urdf"

# Given the params: filename, destination_dir, rename the file and copy it there.
function copy_meshes() {
  local filename=$1
  local destination_dir=$2
  new_name=$(basename "$file")

  # Remove Spider-Leg-Assembly-vXYZ
  new_name=$(echo $new_name | perl -pe 's/Spider-Leg-Assembly-v[0-9]+_//')

  # Shorten motor name and remove version number
  new_name=$(echo $new_name | perl -pe 's/_GIM6010-8-v[0-9]+_Motor/_Motor/')

  cp "$file" "$destination_dir/$new_name"
}

for file in $EXPORT_DIR/urdf/SpiderBody/meshes/*.stl; do
  copy_meshes $file "./meshes/urdf"
done

echo "Done"
