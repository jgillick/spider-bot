# Spider Robot Locomotion in Genesis

This directory contains the Genesis implementation for training an 8-legged spider robot using reinforcement learning.

## Installation

### Local requirements

Create a Python 3.11 virtual environment.

```bash
pyenv virtualenv 3.11 spider-genesis
pyenv activate spider-genesis
```

Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run training

```bash
python ./train.py
```

## Play 

You can play the trained agent with a Logitech gamepad.

First, you need to make sure that HIDAPI is installed on your machine.
https://github.com/libusb/hidapi?tab=readme-ov-file#installing-hidapi

Then:

```bash
python ./play_joystick.py ./logs/<TRAINING DIR>
```

### Troubleshooting

If you have trouble connecting to the gamepad on linux, you might need to update the udev rules:

Create the file: `/etc/udev/rules.d/100-hidapi.rules`

```
SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="c216", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl", SYMLINK+="logitech_f310%n"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c216", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
SUBSYSTEM=="usb", ATTR{idVendor}=="046d", ATTR{idProduct}=="c219", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl", SYMLINK+="logitech_f710%n"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c219", MODE="0660", GROUP="plugdev", TAG+="uaccess", TAG+="udev-acl"
```

Then

```bash
sudo chmod 644 /etc/udev/rules.d/00-hidapi.rules
sudo udevadm control --reload-rules
```