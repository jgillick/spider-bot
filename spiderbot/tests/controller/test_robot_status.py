"""
Tests for hardware/controller/tools/robot_status.py: the GATT packet layouts and the owner
address byte order, which must match what the firmware expects (see firmware/protocol.py).
"""

import asyncio
import os
import struct
import sys
from unittest.mock import AsyncMock, MagicMock, patch

if "bleak" not in sys.modules:
    try:
        import bleak  # noqa: F401
    except ImportError:
        sys.modules["bleak"] = MagicMock()

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "hardware", "controller", "tools")
)

import robot_status  # noqa: E402
from robot_status import (  # noqa: E402
    ADDR_PUBLIC,
    ADDR_RANDOM_STATIC,
    OWNER_UUID,
    ROBOT_STATUS_UUID,
    RobotStatusLink,
)


def make_link():
    link = RobotStatusLink("AA:BB:CC:DD:EE:FF")
    link._client = MagicMock()
    link._client.write_gatt_char = AsyncMock()
    return link


def written(link):
    uuid, data = link._client.write_gatt_char.call_args.args[:2]
    return uuid, data


def test_owner_address_is_sent_lsb_first_with_type():
    link = make_link()
    asyncio.run(link.set_owner("AA:BB:CC:DD:EE:FF", ADDR_RANDOM_STATIC))
    uuid, data = written(link)
    assert uuid == OWNER_UUID
    assert data == bytes([0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, ADDR_RANDOM_STATIC])


def test_robot_status_packet():
    link = make_link()
    asyncio.run(link.send(72, flags=0x01, mode=3))
    uuid, data = written(link)
    assert uuid == ROBOT_STATUS_UUID
    assert data == bytes([72, 0x01, 3, 0])   # reserved byte must be 0 (firmware relies on it)


def test_robot_status_unknown_and_clamped():
    link = make_link()
    asyncio.run(link.send(None))
    assert written(link)[1][0] == 255
    asyncio.run(link.send(150))
    assert written(link)[1][0] == 100
    asyncio.run(link.send(-5))
    assert written(link)[1][0] == 0


def test_decode_controller_status():
    raw = struct.pack("<BBHB", 255, 1, 4100, 2)
    assert RobotStatusLink._decode(raw) == {
        "battery_pct": None,
        "charging": True,
        "battery_mv": 4100,
        "state": "connected",
    }


def test_local_adapter_address_parses_bluetoothctl():
    out = "Controller DC:A6:32:01:02:03 (public)\n\tName: pi\n"
    with patch.object(robot_status, "_bluetoothctl", return_value=out):
        assert robot_status.local_adapter_address() == ("DC:A6:32:01:02:03", ADDR_PUBLIC)
    out = "Controller DC:A6:32:01:02:03 (random)\n"
    with patch.object(robot_status, "_bluetoothctl", return_value=out):
        assert robot_status.local_adapter_address()[1] == ADDR_RANDOM_STATIC


def test_known_controller_matches_by_name():
    out = "Device 11:22:33:44:55:66 Some Mouse\nDevice AA:BB:CC:DD:EE:FF SpiderBot Controller\n"
    with patch.object(robot_status, "_bluetoothctl", return_value=out):
        assert robot_status.known_controller() == "AA:BB:CC:DD:EE:FF"
