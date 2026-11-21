"""BLE side of the controller: HID gamepad, robot-status GATT channel, pairing policy.

Pairing policy (see ../DESIGN.md "Single host, pairing only in pairing mode"):
  * Every HID / GATT attribute needs LESC_ENCRYPT_WITH_MITM. Creating one switches the
    stack to DisplayYesNo + MITM, so *every* new pairing is a numeric comparison that this
    code answers with confirm_pairing(accept=<in pairing mode>).
  * Bonded hosts reconnect with their stored (authenticated) keys; no prompt.
  * A link that isn't authenticated within AUTH_TIMEOUT_MS is dropped. That also catches
    "Just Works" peers, which can encrypt but never authenticate.
  * Only one authenticated link is kept.

Link.poll() runs once per main-loop tick and does all of the above, plus advertising and the
GATT side channel.
"""
import struct

import _bleio
from adafruit_ble.advertising import Advertisement
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.characteristics import Attribute, Characteristic, StructCharacteristic
from adafruit_ble.services import Service
from adafruit_ble.services.standard import BatteryService
from adafruit_ble.services.standard import hid as _hid
from adafruit_ble.uuid import StandardUUID, VendorUUID
from adafruit_ticks import ticks_add, ticks_diff

import config
from protocol import (
    BATTERY_UNKNOWN,
    STATE_ADVERTISING,
    STATE_CONNECTED,
    STATE_IDLE,
    STATE_PAIRING,
)

SECURE = Attribute.LESC_ENCRYPT_WITH_MITM
_REPORT_REFERENCE_UUID = _bleio.UUID(0x2908)
APPEARANCE_GAMEPAD = 964

# --- HID ------------------------------------------------------------------------------
# One input report, ID 1, 9 bytes: X, Y, Z, Rz as int16 (-32767..32767), then
# button 1 (L3) and button 2 (deadman) + 6 bits padding.
# SDL mapping (added host-side): leftx:a0, lefty:a1, rightx:a2, righty:a3,
# leftstick:b0, lefttrigger:b1.
GAMEPAD_DESCRIPTOR = bytes((
    0x05, 0x01,        # Usage Page (Generic Desktop)
    0x09, 0x05,        # Usage (Game Pad)
    0xA1, 0x01,        # Collection (Application)
    0x85, 0x01,        #   Report ID (1)
    0x16, 0x01, 0x80,  #   Logical Minimum (-32767)
    0x26, 0xFF, 0x7F,  #   Logical Maximum (32767)
    0x75, 0x10,        #   Report Size (16)
    0x95, 0x04,        #   Report Count (4)
    0x09, 0x30,        #   Usage (X)
    0x09, 0x31,        #   Usage (Y)
    0x09, 0x32,        #   Usage (Z)
    0x09, 0x35,        #   Usage (Rz)
    0x81, 0x02,        #   Input (Data, Var, Abs)
    0x05, 0x09,        #   Usage Page (Button)
    0x19, 0x01,        #   Usage Minimum (1)
    0x29, 0x02,        #   Usage Maximum (2)
    0x15, 0x00,        #   Logical Minimum (0)
    0x25, 0x01,        #   Logical Maximum (1)
    0x75, 0x01,        #   Report Size (1)
    0x95, 0x02,        #   Report Count (2)
    0x81, 0x02,        #   Input (Data, Var, Abs)
    0x75, 0x06,        #   Report Size (6)
    0x95, 0x01,        #   Report Count (1)
    0x81, 0x03,        #   Input (Const)  padding
    0xC0,              # End Collection
))
REPORT_FMT = "<hhhhB"


# adafruit_ble's HID classes hard-code ENCRYPT_NO_MITM permissions with no option to change
# them, so the two classes below replace those pieces. They copy adafruit_ble 9.0.x internals
# (ReportIn.__init__, and HIDService._init_devices looking ReportIn up as a module global);
# re-check both after upgrading the library.
class _SecureReportIn(_hid.ReportIn):
    """adafruit_ble's ReportIn with MITM-protected permissions."""

    def __init__(self, service, report_id, usage_page, usage, *, max_length):
        # Deliberately not calling super(): it hard-codes ENCRYPT_NO_MITM.
        self._characteristic = _bleio.Characteristic.add_to_service(
            service.bleio_service,
            self.uuid.bleio_uuid,
            properties=Characteristic.READ | Characteristic.NOTIFY,
            read_perm=SECURE,
            write_perm=Attribute.NO_ACCESS,
            max_length=max_length,
            fixed_length=True,
        )
        self._report_id = report_id
        self.usage_page = usage_page
        self.usage = usage
        _bleio.Descriptor.add_to_characteristic(
            self._characteristic,
            _REPORT_REFERENCE_UUID,
            read_perm=SECURE,
            write_perm=Attribute.NO_ACCESS,
            initial_value=struct.pack("<BB", report_id, 1),   # 1 = input report
        )


class SecureHIDService(_hid.HIDService):
    # No boot keyboard on a gamepad.
    boot_keyboard_in = None
    boot_keyboard_out = None

    hid_information = Characteristic(
        uuid=StandardUUID(0x2A4A),
        properties=Characteristic.READ,
        read_perm=SECURE,
        write_perm=Attribute.NO_ACCESS,
        initial_value=b"\x01\x01\x00\x02",
    )
    report_map = Characteristic(
        uuid=StandardUUID(0x2A4B),
        properties=Characteristic.READ,
        read_perm=SECURE,
        write_perm=Attribute.NO_ACCESS,
        fixed_length=True,
    )

    def __init__(self, hid_descriptor):
        # HIDService._init_devices looks ReportIn up as a module global.
        original = _hid.ReportIn
        _hid.ReportIn = _SecureReportIn
        try:
            super().__init__(hid_descriptor)
        finally:
            _hid.ReportIn = original


# --- robot-status side channel -------------------------------------------------------------
def _uuid(n):
    return VendorUUID("5b1d%04x-9c7e-4f21-8b6a-2f3a1c0d4e01" % n)


# There's no "on write" callback, so the robot-status characteristic is used like a mailbox:
# after reading a write, the firmware overwrites it locally with this placeholder. Any value
# the host writes (always reserved byte = 0) then looks different, even if the host sends the
# same status twice, which is what keeps the "robot reported recently" timer fresh.
_CONSUMED_MARK = 0xA5
_ROBOT_STATUS_CONSUMED = bytes((BATTERY_UNKNOWN, 0, 0, _CONSUMED_MARK))


class LinkService(Service):
    """Robot status in, controller status out, owner address in. Layouts: ../DESIGN.md."""

    uuid = _uuid(0x0001)
    robot_status = Characteristic(
        uuid=_uuid(0x0002),
        properties=Characteristic.WRITE | Characteristic.WRITE_NO_RESPONSE,
        read_perm=Attribute.NO_ACCESS,
        write_perm=SECURE,
        max_length=4,
        fixed_length=True,
        initial_value=_ROBOT_STATUS_CONSUMED,
    )
    controller_status = StructCharacteristic(
        "<BBHB",
        uuid=_uuid(0x0003),
        properties=Characteristic.READ | Characteristic.NOTIFY,
        read_perm=SECURE,
        write_perm=Attribute.NO_ACCESS,
        initial_value=(BATTERY_UNKNOWN, 0, 0, STATE_IDLE),
    )
    owner = Characteristic(
        uuid=_uuid(0x0004),
        properties=Characteristic.WRITE,
        read_perm=Attribute.NO_ACCESS,
        write_perm=SECURE,
        max_length=7,
        fixed_length=False,
        initial_value=b"",
    )


# What we're currently advertising.
_ADV_NONE = 0
_ADV_OPEN = 1        # anyone can see and connect (still can't pair outside pairing mode)
_ADV_DIRECTED = 2    # only the stored owner address


class Link:
    def __init__(self, store):
        if not _bleio.adapter.enabled:
            _bleio.adapter.enabled = True
        _bleio.adapter.name = config.BLE_NAME
        self.store = store

        self.hid = SecureHIDService(GAMEPAD_DESCRIPTOR)
        self._gamepad = self.hid.devices[0]
        self.battery = BatteryService()
        self.svc = LinkService()

        self._adv = ProvideServicesAdvertisement(self.hid)
        self._adv.appearance = APPEARANCE_GAMEPAD
        scan = Advertisement()
        scan.complete_name = config.BLE_NAME
        self._adv_bytes = bytes(self._adv)
        self._scan_bytes = bytes(scan)
        self._adv_mode = _ADV_NONE

        self._report = bytearray(struct.calcsize(REPORT_FMT))
        self.pairing = False
        self._pair_until = 0
        self._first_seen = {}
        self._dropping = []       # links we've told to disconnect that haven't gone yet
        self.conn = None          # the authenticated connection, if any

        # Last robot status received over GATT
        self.robot_pct = None
        self.robot_flags = 0
        self._robot_at = None     # when it arrived (ticks_ms), None if not since connecting

    # -- state ------------------------------------------------------------------------
    @property
    def connected(self):
        return self.conn is not None

    @property
    def state(self):
        """One of the protocol STATE_* values."""
        if self.pairing:
            return STATE_PAIRING
        if self.conn is not None:
            return STATE_CONNECTED
        if self._adv_mode != _ADV_NONE:
            return STATE_ADVERTISING
        return STATE_IDLE

    def robot_status_fresh(self, now):
        return (
            self.conn is not None
            and self._robot_at is not None
            and ticks_diff(now, self._robot_at) < config.ROBOT_STATUS_TIMEOUT_MS
        )

    # -- pairing ----------------------------------------------------------------------
    def enter_pairing(self, now):
        """Erase bonds and the stored owner, then accept one new pairing."""
        # disconnect() only *starts* the disconnect: these links still read as connected and
        # authenticated for a moment, and must not be mistaken for the new pairing.
        self._dropping = [c for c in _bleio.adapter.connections if c.connected]
        for c in self._dropping:
            c.disconnect()
        self._stop_adv()
        _bleio.adapter.erase_bonding()
        self.store.clear_owner()
        self.conn = None
        self._first_seen = {}
        self.pairing = True
        self._pair_until = ticks_add(now, config.PAIR_WINDOW_MS)
        print("pairing mode: bonds erased, advertising openly")

    def _answer_pairing_request(self, c):
        """Accept a numeric-comparison request in pairing mode, reject it otherwise."""
        code = c.numeric_comparison
        if code is None:
            return
        print("pairing request %06d -> %s" % (code, "accept" if self.pairing else "reject"))
        try:
            c.confirm_pairing(self.pairing)
        except Exception as e:  # expired between the read and the answer
            print("confirm_pairing:", e)

    def _service_connections(self, now):
        """Answer pairing requests, keep one authenticated host, drop everything else."""
        if self._dropping:
            self._dropping = [c for c in self._dropping if c.connected]
        authed = None
        seen = {}   # connection -> when we first saw it
        for c in _bleio.adapter.connections:
            if not c.connected or c in self._dropping:
                continue
            t0 = self._first_seen.get(c, now)
            seen[c] = t0
            self._answer_pairing_request(c)
            if c.authenticated:
                if authed is None:
                    authed = c
                    if self.pairing:
                        self.pairing = False
                        print("paired")
                else:
                    c.disconnect()          # single host
                continue
            limit = config.AUTH_TIMEOUT_PAIRING_MS if self.pairing else config.AUTH_TIMEOUT_MS
            if ticks_diff(now, t0) > limit:
                print("dropping unauthenticated link")
                c.disconnect()
        self._first_seen = seen
        if authed is not self.conn:
            self.conn = authed
            self._robot_at = None
            print("host connected" if authed else "host disconnected")

    # -- advertising -------------------------------------------------------------------
    def _stop_adv(self):
        if self._adv_mode != _ADV_NONE:
            try:
                _bleio.adapter.stop_advertising()
            except Exception as e:
                print("stop_advertising:", e)
            self._adv_mode = _ADV_NONE

    def _want_adv(self):
        if self.conn is not None:
            return _ADV_NONE
        if self.pairing or not config.DIRECTED_ADVERTISING or self.store.owner is None:
            return _ADV_OPEN
        return _ADV_DIRECTED

    def _service_advertising(self):
        """Start, stop or switch advertising to match _want_adv()."""
        want = self._want_adv()
        adapter = _bleio.adapter
        if want != self._adv_mode:
            self._stop_adv()
        elif want != _ADV_NONE and not adapter.advertising:
            self._adv_mode = _ADV_NONE      # timed out or a connection took it; re-arm
        if want == _ADV_NONE or self._adv_mode == want:
            return
        try:
            if want == _ADV_OPEN:
                adapter.start_advertising(
                    self._adv_bytes, scan_response=self._scan_bytes,
                    connectable=True, interval=0.1,
                )
            else:
                addr, addr_type = self.store.owner
                # Low-duty directed advertising (timeout > 1 s); re-armed when it expires.
                adapter.start_advertising(
                    b"", scan_response=b"", connectable=True, interval=0.1, timeout=30,
                    directed_to=_bleio.Address(addr, addr_type),
                )
            self._adv_mode = want
        except Exception as e:
            print("advertising failed:", e)

    # -- GATT side channel ------------------------------------------------------------------
    def _service_gatt(self, now):
        raw = self.svc.robot_status
        if raw is not None and len(raw) == 4 and raw[3] != _CONSUMED_MARK:
            pct, flags, _mode, _reserved = struct.unpack("<BBBB", raw)
            self.robot_pct = None if pct > 100 else pct
            self.robot_flags = flags
            self._robot_at = now
            self.svc.robot_status = _ROBOT_STATUS_CONSUMED
        owner = self.svc.owner
        if owner and len(owner) >= 6:
            addr_type = owner[6] if len(owner) > 6 else _bleio.Address.PUBLIC
            self.store.save_owner(bytes(owner[:6]), addr_type)
            self.svc.owner = b""
            print("owner address stored")

    # -- public ------------------------------------------------------------------------
    def poll(self, now):
        if self.pairing and ticks_diff(now, self._pair_until) > 0:
            self.pairing = False
            print("pairing window closed")
        self._service_connections(now)
        self._service_advertising()
        self._service_gatt(now)

    def send(self, axes, buttons):
        if self.conn is None:
            return
        struct.pack_into(REPORT_FMT, self._report, 0, axes[0], axes[1], axes[2], axes[3], buttons)
        try:
            self._gamepad.send_report(self._report)
        except Exception as e:
            print("send_report:", e)

    def publish_battery(self, pct, charging, mv):
        """pct: 0-100 or None if unknown. Updates the standard Battery Service and our status."""
        self.battery.level = 0 if pct is None else pct
        self.svc.controller_status = (
            BATTERY_UNKNOWN if pct is None else pct,
            1 if charging else 0,
            max(0, min(65535, mv)),
            self.state,
        )

    def shutdown(self):
        for c in _bleio.adapter.connections:
            if c.connected:
                c.disconnect()
        self._stop_adv()
        _bleio.adapter.enabled = False
