import time
import argparse

from .base import BaseGamepad


class DebugGamepad(BaseGamepad):
    """
    This is just used to output the HID data to the console so you can implement a new gamepad class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_command(self, data):
        print(data)


if __name__ == "__main__":
    """Run from the CLI with python -m genesis_forge.managers.command.gamepads.debug --help"""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-v", "--vender_id", type=str)
    parser.add_argument("-p", "--product_id", type=str)
    args = parser.parse_args()

    vendor_id = int(args.vender_id, 16)
    product_id = int(args.product_id, 16)

    gamepad = DebugGamepad(vendor_id=vendor_id, product_id=product_id)
    while True:
        # print(gamepad.get_command())
        time.sleep(0.1)
