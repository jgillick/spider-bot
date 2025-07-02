# Robot Model

This holds the MuJoCo robot configuration file and 3D meshes.

To generate these files:

1. Using this [Fusion360 model](https://a360.co/4ePgZ3j)
2. The model is exported to MuJoCo format from Fusion360 via [this fork](https://github.com/bionicdl-sustech/ACDC4Robot/pull/9) of the [ACDC4Robot plugin](https://github.com/bionicdl-sustech/ACDC4Robot) to simulate/robot/export/SpiderBody.
3. Finally run `process_export.sh` to make additional changes that are necessary for training.
