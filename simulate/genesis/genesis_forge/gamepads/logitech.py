"""Logitech F310/F710 Gamepad class that uses HID under the hood.

Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/a873d53765a4c83572cf44fa74768ab62ceb7be1/mujoco_playground/experimental/sim2sim/gamepad_reader.py.
"""

import threading
import time
from enum import Enum
import argparse

from .base import BaseGamepad


class LogitechGamepadProduct(Enum):
    F310 = 2
    F710 = 1


LOGITECH_GAMEPADS = {
    "F710": {
        "product_id": 0xC219,
        "lin_x_axis": 1,
        "lin_y_axis": 2,
        "ang_z_axis": 3,
    },
    "F310": {
        "product_id": 0xC216,
        "lin_x_axis": 0,
        "lin_y_axis": 1,
        "ang_z_axis": 2,
    },
}

VENDOR_ID = 0x046D


class LogitechGamepad(BaseGamepad):
    """Implementation for Logitech gamepads."""

    def __init__(self, product: LogitechGamepadProduct, *args, **kwargs):
        self._cfg = LOGITECH_GAMEPADS[product.name]
        super().__init__(
            vendor_id=VENDOR_ID, product_id=self._cfg["product_id"], *args, **kwargs
        )

    def update_command(self, data):
        """
        Process the gamepad data and set vx, vy, and wz to values from -1 to 1
        """
        self.vx = -(data[self._cfg["lin_x_axis"]] - 128) / 128.0
        self.vy = -(data[self._cfg["lin_y_axis"]] - 128) / 128.0
        self.wz = -(data[self._cfg["ang_z_axis"]] - 128) / 128.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-p", "--product", type=str)
    args = parser.parse_args()

    product = LogitechGamepadProduct.F710
    if args.product.upper() == "F310":
        product = LogitechGamepadProduct.F310
    gamepad = LogitechGamepad(product)
    while True:
        print(gamepad.get_command())
        time.sleep(1.0)
