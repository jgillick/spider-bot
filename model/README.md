# Robot Model

The functional 3D model of the robot, converted from Fusion360, and processed for reinforcement learning environments.

To generate these files:

1. Using this [Fusion360 model](https://a360.co/4dRpD23)
2. The model is exported to MuJoCo format from Fusion360 via the [Fusion2Mujoco add-in](https://github.com/jgillick/Fusion2Mujoco) to a sub directory (v1, v2, etc).
3. Finally, update the `<include>` reference to the exported model in [./SpiderBot.xml](./SpiderBot.xml)
