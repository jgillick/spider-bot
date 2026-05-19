# Setup

Before using an actuator, you must calibrate it in odrivetool:

```sh
odrivetool

odrv0.axis0.requested_state = AXIS_STATE_MOTOR_CALIBRATION
dump_errors(odrv0)

odrv0.axis0.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
dump_errors(odrv0)

odrv0.axis0.motor.config.pre_calibrated = 1
odrv0.axis0.encoder.config.pre_calibrated = 1
odrv0.save_configuration()
```

# Reset

If you brick your actuator, you can clear the configuration:

```sh
odrivetool

odrv0.erase_configuration()
odrv0.reboot()
```

Now re-calibrate the motor (see Setup section)
