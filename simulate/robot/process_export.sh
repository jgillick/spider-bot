#!/bin/bash
set -e

EXPORT_DIR="./export/SpiderBody"

echo "Processing XML..."

# Create two versions of the XML: one with the ground plane and one without
python update_xml.py SpiderBotNoEnv.xml
python update_xml.py --ground --light SpiderBot.xml

###
# Move mesh files, if they exist
echo "Updating meshes..."
if [ -d "./meshes" ]; then
  rm -r ./meshes/
fi
mkdir "./meshes"
for file in $EXPORT_DIR/meshes/*.stl; do
  new_name=$(basename "$file")

  # Remove Spider-Leg-Assembly-vXYZ
  new_name=$(echo $new_name | perl -pe 's/Spider-Leg-Assembly-v[0-9]+_//')

  # Shorten motor name and remove version number
  new_name=$(echo $new_name | perl -pe 's/_GIM6010-8-v[0-9]+_Motor/_Motor/')

  cp "$file" "./meshes/$new_name"
done

echo "Done"
