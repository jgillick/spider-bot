#!/usr/bin/env python3
"""
Push the robot's battery / state to the controller's LED bar over the GATT side channel,
read the controller's own battery back, and register this host as the controller's owner.

The controller is already connected to this machine as a Bluetooth HID gamepad (that's how
SDL sees it). BlueZ lets a second process open GATT on the same connection, so this runs
alongside genesis-forge without touching the HID path.

    pip install bleak
    ./robot_status.py --set-owner                  # once, right after pairing (Linux)
    ./robot_status.py --battery 72                 # one-shot
    ./robot_status.py --watch                      # print controller status, send a demo ramp
    ./robot_status.py --address AA:BB:CC:DD:EE:FF  # skip discovery

In the robot's own code, import `RobotStatusLink` and call `await link.send(pct, flags)` from
wherever the pack voltage is read. Send at least every few seconds: the controller blanks
the bar when it hasn't heard from the robot for 5 s.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import struct
import subprocess
import sys

from bleak import BleakClient, BleakScanner

SVC_UUID = "5b1d0001-9c7e-4f21-8b6a-2f3a1c0d4e01"
ROBOT_STATUS_UUID = "5b1d0002-9c7e-4f21-8b6a-2f3a1c0d4e01"   # write: battery_pct, flags, mode, reserved
CTRL_STATUS_UUID = "5b1d0003-9c7e-4f21-8b6a-2f3a1c0d4e01"    # notify: battery_pct, charging, mv(u16), state
OWNER_UUID = "5b1d0004-9c7e-4f21-8b6a-2f3a1c0d4e01"          # write: host BT address (LSB first) + type

FLAG_FAULT = 0x01
FLAG_POLICY_RUNNING = 0x02
FLAG_ESTOP = 0x04

BATTERY_UNKNOWN = 255     # same values as firmware/protocol.py
ADDR_PUBLIC = 0
ADDR_RANDOM_STATIC = 1

DEVICE_NAME = "SpiderBot Controller"   # firmware config.BLE_NAME
LINK_STATES = {0: "idle", 1: "advertising", 2: "connected", 3: "pairing"}

_MAC = re.compile(r"([0-9A-F]{2}(?::[0-9A-F]{2}){5})", re.I)


def _bluetoothctl(*args: str) -> str:
    if not shutil.which("bluetoothctl"):
        return ""
    try:
        return subprocess.run(
            ["bluetoothctl", *args], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def known_controller() -> str | None:
    """A connected or paired controller from BlueZ. Scanning can't see it while it's
    connected, because it stops advertising."""
    for which in ("Connected", "Paired"):
        for line in _bluetoothctl("devices", which).splitlines():
            m = _MAC.search(line)
            if m and DEVICE_NAME in line:
                return m.group(1)
    return None


async def find_controller(timeout: float = 8.0) -> str | None:
    """Address of the controller: from BlueZ if it's paired, else from a scan."""
    addr = known_controller()
    if addr:
        return addr
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
    for dev, adv in devices.values():
        name = adv.local_name or dev.name or ""
        if name == DEVICE_NAME or SVC_UUID in (adv.service_uuids or []):
            return dev.address
    return None


def local_adapter_address() -> tuple[str, int] | None:
    """This host's Bluetooth address and type, from `bluetoothctl show` (Linux/BlueZ)."""
    for line in _bluetoothctl("show").splitlines():
        if line.startswith("Controller"):
            m = _MAC.search(line)
            if m:
                kind = ADDR_RANDOM_STATIC if "(random)" in line else ADDR_PUBLIC
                return m.group(1), kind
    return None


class RobotStatusLink:
    """Small async wrapper the robot can keep open for the whole session."""

    def __init__(self, address: str):
        self.address = address
        self._client: BleakClient | None = None

    async def __aenter__(self):
        self._client = BleakClient(self.address)
        await self._client.connect()
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.disconnect()

    async def send(self, battery_pct: int | None, flags: int = 0, mode: int = 0) -> None:
        """battery_pct 0-100, or None if unknown."""
        pct = BATTERY_UNKNOWN if battery_pct is None else max(0, min(100, int(battery_pct)))
        pkt = struct.pack("<BBBB", pct, flags & 0xFF, mode & 0xFF, 0)
        await self._client.write_gatt_char(ROBOT_STATUS_UUID, pkt, response=False)

    async def set_owner(self, address: str, address_type: int = ADDR_PUBLIC) -> None:
        """Store `address` (AA:BB:...) as the only host the controller advertises to."""
        raw = bytes(int(b, 16) for b in reversed(address.split(":")))
        await self._client.write_gatt_char(OWNER_UUID, raw + bytes([address_type]), response=True)

    async def read_controller(self) -> dict:
        raw = await self._client.read_gatt_char(CTRL_STATUS_UUID)
        return self._decode(raw)

    async def subscribe(self, cb) -> None:
        await self._client.start_notify(CTRL_STATUS_UUID, lambda _h, d: cb(self._decode(d)))

    @staticmethod
    def _decode(raw: bytes) -> dict:
        pct, chg, mv, state = struct.unpack("<BBHB", bytes(raw[:5]))
        return {
            "battery_pct": None if pct == BATTERY_UNKNOWN else pct,
            "charging": bool(chg),
            "battery_mv": mv,
            "state": LINK_STATES.get(state, state),
        }


def resolve_owner(arg: str) -> tuple[str, int] | None:
    """--set-owner value -> (address, type). 'auto' means this host's own adapter."""
    if arg == "auto":
        return local_adapter_address()
    return arg, ADDR_PUBLIC          # an explicit address is assumed to be public


async def demo_ramp(link: RobotStatusLink, flags: int) -> None:
    """Send 100, 90, ... 0 % every 2 s, forever, so you can watch the bar step down."""
    pct = 100
    while True:
        await link.send(pct, flags)
        await asyncio.sleep(2.0)
        pct = 100 if pct <= 0 else pct - 10


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", help="controller BLE address (default: discover)")
    ap.add_argument("--battery", type=int, help="robot battery %% to send once (-1 = unknown)")
    ap.add_argument("--fault", action="store_true", help="set the fault flag")
    ap.add_argument("--watch", action="store_true", help="stay connected, print controller status, ramp demo")
    ap.add_argument("--set-owner", nargs="?", const="auto", metavar="ADDR",
                    help="register this host (or a public ADDR) for directed advertising")
    args = ap.parse_args()

    addr = args.address or await find_controller()
    if not addr:
        print("controller not found (is it on and paired to this machine?)", file=sys.stderr)
        return 1

    flags = FLAG_FAULT if args.fault else 0
    async with RobotStatusLink(addr) as link:
        print("controller:", await link.read_controller())
        if args.set_owner:
            owner = resolve_owner(args.set_owner)
            if not owner:
                print("couldn't read this host's address (bluetoothctl show); pass it explicitly",
                      file=sys.stderr)
                return 1
            address, kind = owner
            await link.set_owner(address, kind)
            print(f"owner set to {address} ({'random' if kind == ADDR_RANDOM_STATIC else 'public'})")
        if args.battery is not None:
            pct = None if args.battery < 0 else args.battery
            await link.send(pct, flags)
            print(f"sent battery={pct} flags={flags:#04x}")
        if args.watch:
            await link.subscribe(lambda s: print("controller:", s))
            await demo_ramp(link, flags)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
