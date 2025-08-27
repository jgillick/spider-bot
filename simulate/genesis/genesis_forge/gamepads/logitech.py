"""Logitech F310/F710 Gamepad configuration class."""

import time
from enum import Enum
import argparse

from .base import BaseGamepad, Button


class LogitechGamepadProduct(Enum):
    F310 = 2
    F710 = 1


VENDOR_ID = 0x046D

LOGITECH_GAMEPADS = {
    "F710": {
        "product_id": 0xC219,
        "mapping": [
            {"axis": 0, "data": 1},
            {"axis": 1, "data": 2},
            {"axis": 2, "data": 3},
            {"axis": 3, "data": 4},
            # TODO: Need to map all buttons
        ],
    },
    "F310": {
        "product_id": 0xC216,
        "mapping": [
            {"axis": 0, "data": 0},
            {"axis": 1, "data": 1},
            {"axis": 2, "data": 2},
            {"axis": 3, "data": 3},
            {"button": Button.A, "data": 4, "bitmask": 32},
            {"button": Button.B, "data": 4, "bitmask": 64},
            {"button": Button.X, "data": 4, "bitmask": 16},
            {"button": Button.Y, "data": 4, "bitmask": 128},
            {"button": Button.LB, "data": 5, "bitmask": 1},
            {"button": Button.RB, "data": 5, "bitmask": 2},
            {"button": Button.LT, "data": 5, "bitmask": 4},
            {"button": Button.RT, "data": 5, "bitmask": 8},
            {"button": Button.BACK, "data": 5, "bitmask": 16},
            {"button": Button.START, "data": 5, "bitmask": 32},
            {"button": Button.MODE, "data": 6, "bitmask": 8},
            {"button": Button.LEFT_JOYSTICK, "data": 5, "bitmask": 65},
            {"button": Button.RIGHT_JOYSTICK, "data": 5, "bitmask": 128},
        ],
    },
}


class LogitechGamepad(BaseGamepad):
    """
    Connect to a Logitech gamepad.

    Args:
        product: The product to connect to.
    """

    def __init__(self, product: LogitechGamepadProduct, *args, **kwargs):
        self.cfg = LOGITECH_GAMEPADS[product.name]
        super().__init__(
            vendor_id=VENDOR_ID,
            product_id=self.cfg["product_id"],
            mapping=self.cfg["mapping"],
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-p", "--product", type=str)
    args = parser.parse_args()

    product = LogitechGamepadProduct.F710
    if args.product.upper() == "F310":
        product = LogitechGamepadProduct.F310
    gamepad = LogitechGamepad(product, debug=True)
    while True:
        time.sleep(1.0)
