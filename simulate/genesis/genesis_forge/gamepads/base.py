"""Logitech F310/F710 Gamepad class that uses HID under the hood.

Adapted from: https://github.com/google-deepmind/mujoco_playground/blob/a873d53765a4c83572cf44fa74768ab62ceb7be1/mujoco_playground/experimental/sim2sim/gamepad_reader.py.
"""

import threading

import hid
import numpy as np


class BaseGamepad:
    """Base gamepad class.
    Subclasses should just need to set the vendor_id and product_id and implement the update_command method
    """

    def __init__(
        self,
        vendor_id=0x0000,
        product_id=0x0000,
        vel_scale_lin_x=1.0,
        vel_scale_lin_y=1.0,
        vel_scale_angle_z=1.0,
    ):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._vel_scale_x = vel_scale_lin_x
        self._vel_scale_y = vel_scale_lin_y
        self._vel_scale_z = vel_scale_angle_z

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.is_running = True

        self._device = None

        self.read_thread = threading.Thread(target=self.read_loop, daemon=True)
        self.read_thread.start()

    def update_command(self, data):
        """
        Process the gamepad data and set vx, vy, and wz to values from -1 to 1
        """
        raise NotImplementedError("update_command not implemented")

    def _connect_device(self):
        try:
            self._device = hid.device()
            self._device.open(self._vendor_id, self._product_id)
            self._device.set_nonblocking(True)
            print(
                f"Connected to gamepad {self._device.get_manufacturer_string()} {self._device.get_product_string()}"
            )
            return True
        except IOError as e:
            print(
                f"Error connecting to gamepad 0x{self._vendor_id:04x}:0x{self._product_id:04x}: {e}"
            )
            return False

    def read_loop(self):
        if not self._connect_device():
            self.is_running = False
            return

        while self.is_running:
            try:
                data = self._device.read(64)
                if data:
                    self.update_command(data)
            except Exception as e:
                print(f"Error reading from device: {e}")

        self._device.close()

    def get_command(self):
        return np.array([self.vx, self.vy, self.wz])

    def stop(self):
        self.is_running = False
