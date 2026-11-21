"""Persistent settings in microcontroller.nvm: stick calibration and the owner address.

Layout (little-endian):
    0   2s  magic b"SB"
    2   B   version
    3   B   flags (bit0 calibration valid, bit1 owner valid)
    4   12H per axis (min, centre, max) x 4
    28  6s  owner BT address, LSB first (same order as _bleio.Address.address_bytes)
    34  B   owner address type
"""
import struct

import microcontroller

_FMT = "<2sBB12H6sB"
_SIZE = struct.calcsize(_FMT)
_MAGIC = b"SB"
_VERSION = 1
_F_CAL = 0x01
_F_OWNER = 0x02


class Store:
    def __init__(self):
        self.cal = None          # list of 4 (min, centre, max) tuples, or None
        self.owner = None        # (address_bytes, address_type) or None
        self._load()

    def _load(self):
        try:
            raw = bytes(microcontroller.nvm[0:_SIZE])
            fields = struct.unpack(_FMT, raw)
        except (ValueError, TypeError):
            return
        magic, version, flags = fields[0], fields[1], fields[2]
        if magic != _MAGIC or version != _VERSION:
            return
        if flags & _F_CAL:
            v = fields[3:15]
            self.cal = [(v[i], v[i + 1], v[i + 2]) for i in range(0, 12, 3)]
        if flags & _F_OWNER:
            self.owner = (fields[15], fields[16])

    def _save(self):
        flags = 0
        cal = [0] * 12
        if self.cal:
            flags |= _F_CAL
            cal = [x for triple in self.cal for x in triple]
        owner, owner_type = b"\x00" * 6, 0
        if self.owner:
            flags |= _F_OWNER
            owner, owner_type = self.owner
        raw = struct.pack(_FMT, _MAGIC, _VERSION, flags, *cal, owner, owner_type)
        microcontroller.nvm[0:_SIZE] = raw

    def save_cal(self, cal):
        self.cal = [tuple(int(x) for x in t) for t in cal]
        self._save()

    def save_owner(self, address_bytes, address_type):
        new = (bytes(address_bytes), int(address_type))
        if new != self.owner:
            self.owner = new
            self._save()

    def clear_owner(self):
        if self.owner is not None:
            self.owner = None
            self._save()
