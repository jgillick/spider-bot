"""Logitech F310/F710 Gamepad class that uses HID under the hood.

Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/a873d53765a4c83572cf44fa74768ab62ceb7be1/mujoco_playground/experimental/sim2sim/gamepad_reader.py.
"""

import hid
import threading
from typing import TypedDict, Literal
from enum import Enum
import numpy as np


class Button(Enum):
    A = 1
    B = 2
    X = 3
    Y = 4
    LB = 5
    RB = 6
    LT = 7
    RT = 8
    BACK = 9
    START = 10
    MODE = 11
    LEFT_JOYSTICK = 1
    RIGHT_JOYSTICK = 2


class GamepadMapping(TypedDict):
    """
    Defines the how to extract a value from a gamepad data array.
    """

    data: int
    """The index of the data in the gamepad data array."""

    button: Button
    """If this is a button, what is it's name"""

    axis: int
    """If this is a joystick, what is it's axis number"""

    bitmask: int
    """The bitmask to extract the value from the data."""


class GamepadState:
    """Data from a gamepad."""

    axis_values: list[float] = []
    """The value (-1 to 1) for each axes."""

    buttons: list[str] = []
    """A list of the buttons that are pressed."""

    def axis(self, index: int):
        """
        Get the axis value at an index.
        This is the preferred way to get the axis value, because the axis array will not be filled until the gamepad
        receives input.

        Args:
            index: The index of the axis to get the value of.

        Returns:
            The value of the axis at the index.
        """
        if index >= len(self.axis_values):
            return 0.0
        return self.axis_values[index]

    def __repr__(self):
        return f"GamepadState(axis={self.axis_values}, buttons={self.buttons})"


class BaseGamepad:
    """Base gamepad class.
    Subclasses should just need to set the vendor_id and product_id and implement the update_command method
    """

    def __init__(
        self,
        vendor_id=0x0000,
        product_id=0x0000,
        debug=False,
        mapping: list[GamepadMapping] = [],
    ):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._state = GamepadState()
        self._mapping = mapping
        self._debug = debug

        self.is_running = True
        self._device = None

        self.connect()
        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    @property
    def state(self) -> GamepadState:
        return self._state

    def parse_data(self, data: list[int]) -> GamepadState:
        """Parse gamepad data into a GamepadState object."""
        axis = []
        buttons = []

        for cfg in self._mapping:
            if "data" not in cfg:
                print(f"Warning: {cfg} has no data value")
                continue
            if cfg["data"] >= len(data):
                print(f"Error: {cfg} data is out of range")
                continue
            value = data[cfg["data"]]

            if "button" in cfg:
                if "bitmask" not in cfg:
                    print(f"Warning: {cfg} has no bitmask value")
                    continue
                if value & cfg["bitmask"] != 0:
                    buttons.append(cfg["button"].name)
            elif "axis" in cfg:
                value = -(value - 128) / 128.0
                axis.insert(cfg["axis"], value)

        self._state.axis_values = axis
        self._state.buttons = buttons
        return self._state

    def connect(self):
        try:
            self._device = hid.device()
            self._device.open(self._vendor_id, self._product_id)
            self._device.set_nonblocking(True)
            print(
                f"Connected to gamepad {self._device.get_manufacturer_string()} {self._device.get_product_string()}"
            )
            return True
        except IOError as e:
            raise IOError(
                f"Error connecting to gamepad 0x{self._vendor_id:04x}:0x{self._product_id:04x}: {e}"
            )

    def read_loop(self):
        while self.is_running:
            try:
                data = self._device.read(64)
                if data:
                    try:
                        self._state = self.parse_data(data)
                        if self._debug:
                            print(self._state)
                    except Exception as e:
                        print(f"Error parsing data: {e}")
            except Exception as e:
                print(f"Error reading from device: {e}")

        self._device.close()

    def stop(self):
        self.is_running = False
