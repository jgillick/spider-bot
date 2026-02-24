"""
Non-blocking, multi-node CAN bus controller for ODrive motors.

A background thread continuously reads CAN messages and updates per-node
state.  Send methods are non-blocking and may be called from any thread.

Shared protocol constants and enums are imported from protocol and utils.

Example
-------
    bus = CanMotorController(channel="/dev/ttyACM0")
    bus.register_node(1)
    bus.register_node(2)
    bus.start()

    bus.send_mit_control(1, position=0.5)
    state = bus.get_state(1)
    print(state.position, state.is_connected)

    stale = bus.get_stale_nodes()
    bus.stop()
"""

from __future__ import annotations

import can
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .protocol import (
    MOTOR_TO_RAD,
    MOTOR_UNITS_PER_REV,
    RAD_TO_MOTOR,
    MIT_POS_MIN,
    MIT_POS_MAX,
    MIT_VEL_MIN,
    MIT_VEL_MAX,
    MIT_KP_MIN,
    MIT_KP_MAX,
    MIT_KD_MIN,
    MIT_KD_MAX,
    MIT_TORQUE_MIN,
    MIT_TORQUE_MAX,
    HEARTBEAT_TIMEOUT,
    ENCODER_OFFSET_ENDPOINT_ID,
    ENCODER_POSITION_ESTIMATE_ID,
    AxisState,
    ControlMode,
    InputMode,
    CANCommand,
)
from .utils import make_can_id, rad_to_rev




class CanMotorController:
    """
    Non-blocking, multi-node CAN bus controller.

    Register motor nodes, then call :meth:`start` to open the bus and begin
    background reception.  Query the latest state for any node with
    :meth:`get_state`, and send commands with the ``send_*`` family of
    methods.

    Parameters
    ----------
    channel : str
        CAN adapter path (e.g. ``"/dev/ttyACM0"``).
    bitrate : int
        CAN bus bitrate (default 500 000).
    interface : str
        python-can interface type (default ``"slcan"``).
    heartbeat_timeout : float
        Seconds without a heartbeat before a node is considered stale.
    """

    def __init__(
        self,
        channel: str,
        bitrate: int = 500_000,
        interface: str = "slcan",
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
    ):
        self._channel = channel
        self._bitrate = bitrate
        self._interface = interface
        self._heartbeat_timeout = heartbeat_timeout

        self._bus: Optional[can.Bus] = None
        self._nodes: dict[int, NodeData] = {}
        self._nodes_lock = threading.Lock()

        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Open the CAN bus and start the background reader thread."""
        if self._bus is not None:
            return

        self._bus = can.interface.Bus(
            channel=self._channel,
            interface=self._interface,
            bitrate=self._bitrate,
        )
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="can-reader"
        )
        self._reader_thread.start()

    def stop(self) -> None:
        """Stop the background reader and close the CAN bus."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- Node registration ------------------------------------------------

    def register_node(self, node_id: int) -> None:
        """Add a node to the bus."""
        with self._nodes_lock:
            if node_id not in self._nodes:
                self._nodes[node_id] = NodeData(node_id)

    def unregister_node(self, node_id: int) -> None:
        """Remove a node from the bus."""
        with self._nodes_lock:
            self._nodes.pop(node_id, None)

    @property
    def registered_nodes(self) -> list[int]:
        """Return a copy of all registered node IDs."""
        with self._nodes_lock:
            return list(self._nodes.keys())

    # -- State queries ----------------------------------------------------

    def get_state(self, node_id: int) -> NodeSnapshot:
        """
        Return the latest state snapshot for *node_id*.

        Raises :class:`KeyError` if the node has not been registered.
        """
        with self._nodes_lock:
            node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} is not registered")
        return node.snapshot(self._heartbeat_timeout)

    def get_stale_nodes(self) -> list[int]:
        """Return node IDs that have not received a heartbeat within the timeout."""
        now = time.monotonic()
        stale: list[int] = []
        with self._nodes_lock:
            nodes = list(self._nodes.values())
        for nd in nodes:
            with nd.lock:
                if (
                    nd.last_heartbeat is None
                    or (now - nd.last_heartbeat) >= self._heartbeat_timeout
                ):
                    stale.append(nd.node_id)
        return stale

    def has_errors(self) -> bool:
        """Return ``True`` if any registered node has a non-zero axis error."""
        with self._nodes_lock:
            nodes = list(self._nodes.values())
        for nd in nodes:
            with nd.lock:
                if nd.axis_error != 0:
                    return True
        return False

    def get_axis_errors(self) -> dict[int, int]:
        """Return a mapping of node ID → axis error code for all registered nodes."""
        result: dict[int, int] = {}
        with self._nodes_lock:
            nodes = list(self._nodes.values())
        for nd in nodes:
            with nd.lock:
                result[nd.node_id] = nd.axis_error
        return result

    # -- Send commands (non-blocking) -------------------------------------

    def _send(self, cmd_id: int, node_id: int, data: bytes = b"") -> None:
        if self._bus is None:
            raise RuntimeError("Bus is not started")
        msg = can.Message(
            arbitration_id=make_can_id(cmd_id, node_id),
            data=data,
            is_extended_id=False,
        )
        self._bus.send(msg)

    def send_mit_control(
        self,
        node_id: int,
        position: float,
        kp: float = 52.0,
        kd: float = 1.2,
        velocity: float = 0.0,
        torque: float = 0.0,
    ) -> None:
        """
        Send an MIT impedance-control command to *node_id*.

        The motor applies::

            torque_out = kp*(position - pos_actual)
                       + kd*(velocity - vel_actual)
                       + torque

        All values are clamped to the protocol's valid ranges.

        Parameters
        ----------
        node_id : int
            Target CAN node.
        position : float
            Target position in radians (-12.5 .. 12.5).
        kp : float
            Position gain (0 .. 500).
        kd : float
            Velocity / damping gain (0 .. 5).
        velocity : float
            Target velocity in rad/s (-65 .. 65).
        torque : float
            Feed-forward torque in Nm (-50 .. 50).
        """
        position = max(MIT_POS_MIN, min(MIT_POS_MAX, position))
        velocity = max(MIT_VEL_MIN, min(MIT_VEL_MAX, velocity))
        kp = max(MIT_KP_MIN, min(MIT_KP_MAX, kp))
        kd = max(MIT_KD_MIN, min(MIT_KD_MAX, kd))
        torque = max(MIT_TORQUE_MIN, min(MIT_TORQUE_MAX, torque))

        pos_int = int((position - MIT_POS_MIN) * 65535 / (MIT_POS_MAX - MIT_POS_MIN))
        vel_int = int((velocity - MIT_VEL_MIN) * 4095 / (MIT_VEL_MAX - MIT_VEL_MIN))
        kp_int = int((kp - MIT_KP_MIN) * 4095 / (MIT_KP_MAX - MIT_KP_MIN))
        kd_int = int((kd - MIT_KD_MIN) * 4095 / (MIT_KD_MAX - MIT_KD_MIN))
        t_int = int(
            (torque - MIT_TORQUE_MIN) * 4095 / (MIT_TORQUE_MAX - MIT_TORQUE_MIN)
        )

        pos_int = max(0, min(0xFFFF, pos_int))
        vel_int = max(0, min(0xFFF, vel_int))
        kp_int = max(0, min(0xFFF, kp_int))
        kd_int = max(0, min(0xFFF, kd_int))
        t_int = max(0, min(0xFFF, t_int))

        data = bytes(
            [
                (pos_int >> 8) & 0xFF,
                pos_int & 0xFF,
                (vel_int >> 4) & 0xFF,
                ((vel_int & 0xF) << 4) | ((kp_int >> 8) & 0xF),
                kp_int & 0xFF,
                (kd_int >> 4) & 0xFF,
                ((kd_int & 0xF) << 4) | ((t_int >> 8) & 0xF),
                t_int & 0xFF,
            ]
        )
        self._send(CANCommand.MIT_CONTROL, node_id, data)

    def send_position(
        self,
        node_id: int,
        position: float,
        vel_ff: float = 0.0,
        torque_ff: float = 0.0,
    ) -> None:
        """
        Send a position setpoint (SET_INPUT_POS) to *node_id*.

        Parameters
        ----------
        node_id : int
            Target CAN node.
        position : float
            Target position in radians.
        vel_ff : float
            Velocity feed-forward in rad/s.
        torque_ff : float
            Torque feed-forward in Nm.
        """
        motor_pos = position * RAD_TO_MOTOR
        vel_ff_units = int(rad_to_rev(vel_ff) / 0.001)
        data = struct.pack("<fhh", motor_pos, vel_ff_units, int(torque_ff / 0.001))
        self._send(CANCommand.SET_INPUT_POS, node_id, data)

    def send_set_state(self, node_id: int, state: AxisState) -> None:
        """Set axis state on *node_id* (e.g. CLOSED_LOOP_CONTROL, IDLE)."""
        data = struct.pack("<I", int(state))
        self._send(CANCommand.SET_AXIS_STATE, node_id, data)

    def send_set_control_mode(
        self, node_id: int, control_mode: ControlMode, input_mode: InputMode
    ) -> None:
        """Set controller mode and input mode on *node_id*."""
        data = struct.pack("<II", int(control_mode), int(input_mode))
        self._send(CANCommand.SET_CONTROLLER_MODE, node_id, data)

    def send_clear_errors(self, node_id: int) -> None:
        """Clear axis errors on *node_id*."""
        self._send(CANCommand.CLEAR_ERRORS, node_id, struct.pack("<B", 0))

    def send_estop(self, node_id: int) -> None:
        """Emergency-stop *node_id*."""
        self._send(CANCommand.ESTOP, node_id)

    def send_save_configuration(self, node_id: int) -> None:
        """Save configuration to flash on *node_id* (motor will reboot)."""
        self._send(CANCommand.SAVE_CONFIGURATION, node_id)

    # -- SDO read / write -------------------------------------------------

    def _get_node(self, node_id: int) -> "NodeData":
        with self._nodes_lock:
            node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} is not registered")
        return node

    def read_float_parameter(
        self, node_id: int, endpoint_id: int, timeout: float = 1.5
    ) -> float:
        """
        Read a float from an SDO endpoint on *node_id*.

        Sends an RxSDO read request and waits (blocks the calling thread,
        **not** the background reader) for the TxSDO response.

        Raises :class:`TimeoutError` if no response arrives within *timeout*
        seconds.
        """
        node = self._get_node(node_id)
        event = threading.Event()

        with node.lock:
            node.sdo_waiters[endpoint_id] = event
            node.sdo_results.pop(endpoint_id, None)

        data = struct.pack("<BHB", 0, endpoint_id & 0xFFFF, 0) + b"\x00\x00\x00\x00"
        self._send(CANCommand.RXSDO, node_id, data)

        if not event.wait(timeout=timeout):
            with node.lock:
                node.sdo_waiters.pop(endpoint_id, None)
            raise TimeoutError(
                f"sdo_read_float node {node_id} endpoint {endpoint_id}: "
                f"no response within {timeout}s"
            )

        with node.lock:
            node.sdo_waiters.pop(endpoint_id, None)
            return node.sdo_results.pop(endpoint_id)

    def write_float_parameter(
        self, node_id: int, endpoint_id: int, value: float
    ) -> None:
        """Write a float to an SDO endpoint on *node_id* (fire-and-forget)."""
        payload = struct.pack("<f", value)
        data = struct.pack("<BHB", 1, endpoint_id & 0xFFFF, 0) + payload
        self._send(CANCommand.RXSDO, node_id, data)

    def save_zero_to_motor(self, node_id: int) -> None:
        """
        Persist the current position as zero on *node_id*.

        Reads the current encoder offset and position estimate via SDO,
        computes the new offset, writes it back, and saves configuration
        to flash.  The motor will reboot after save.
        """
        current_offset = self.read_float_parameter(node_id, ENCODER_OFFSET_ENDPOINT_ID)
        pos_estimate = self.read_float_parameter(node_id, ENCODER_POSITION_ESTIMATE_ID)
        new_offset = (pos_estimate + current_offset) % MOTOR_UNITS_PER_REV
        self.write_float_parameter(node_id, ENCODER_OFFSET_ENDPOINT_ID, new_offset)
        time.sleep(0.05)
        self.send_save_configuration(node_id)

    # -- Background reader ------------------------------------------------

    def _reader_loop(self) -> None:
        """Continuously read from the CAN bus and dispatch to node state."""
        while not self._stop_event.is_set():
            if self._bus is None:
                break
            try:
                msg = self._bus.recv(timeout=0.05)
            except can.CanError:
                continue
            if msg is None:
                continue

            cmd_id = msg.arbitration_id & 0x1F
            recv_node = msg.arbitration_id >> 5

            with self._nodes_lock:
                node = self._nodes.get(recv_node)
            if node is None:
                continue

            with node.lock:
                if cmd_id == CANCommand.HEARTBEAT and len(msg.data) >= 5:
                    node.last_heartbeat = time.monotonic()
                    node.axis_error = struct.unpack("<I", msg.data[0:4])[0]
                    node.axis_state = AxisState(msg.data[4])

                elif cmd_id == CANCommand.MIT_CONTROL and len(msg.data) >= 6:
                    d = msg.data
                    t_int = ((d[4] & 0xF) << 8) | d[5]
                    node.mit_torque = (
                        t_int * (MIT_TORQUE_MAX - MIT_TORQUE_MIN) / 4095
                        + MIT_TORQUE_MIN
                    )

                elif (
                    cmd_id == CANCommand.GET_ENCODER_ESTIMATES
                    and len(msg.data) >= 8
                ):
                    node.position, node.velocity = struct.unpack(
                        "<ff", msg.data
                    )

                elif (
                    cmd_id in (CANCommand.TXSDO, CANCommand.RXSDO)
                    and len(msg.data) >= 8
                ):
                    endpoint_id = struct.unpack("<H", msg.data[1:3])[0]
                    waiter = node.sdo_waiters.get(endpoint_id)
                    if waiter is not None:
                        node.sdo_results[endpoint_id] = struct.unpack(
                            "<f", msg.data[4:8]
                        )[0]
                        waiter.set()


@dataclass(frozen=True)
class NodeSnapshot:
    """Immutable point-in-time snapshot of a motor node's state."""

    node_id: int
    position: float  # radians (raw encoder)
    velocity: float  # rad/s
    axis_state: AxisState
    axis_error: int
    mit_torque: float  # Nm, output-shaft side
    last_heartbeat: Optional[float]  # monotonic timestamp, None if never received
    heartbeat_timeout: float = HEARTBEAT_TIMEOUT

    @property
    def is_connected(self) -> bool:
        """True if a heartbeat was received within the timeout window."""
        if self.last_heartbeat is None:
            return False
        return (time.monotonic() - self.last_heartbeat) < self.heartbeat_timeout


class NodeData:
    """Mutable internal state for a single node, protected by its own lock."""

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.lock = threading.Lock()
        self.position = 0.0  # motor units
        self.velocity = 0.0  # motor units/s
        self.axis_state = AxisState.UNDEFINED
        self.axis_error = 0
        self.mit_torque = 0.0
        self.last_heartbeat: Optional[float] = None

        # SDO request-response: endpoint_id -> (Event, result_value)
        self.sdo_waiters: dict[int, threading.Event] = {}
        self.sdo_results: dict[int, float] = {}

    def snapshot(self, heartbeat_timeout: float) -> NodeSnapshot:
        with self.lock:
            return NodeSnapshot(
                node_id=self.node_id,
                position=self.position * MOTOR_TO_RAD,
                velocity=self.velocity * MOTOR_TO_RAD,
                axis_state=self.axis_state,
                axis_error=self.axis_error,
                mit_torque=self.mit_torque,
                last_heartbeat=self.last_heartbeat,
                heartbeat_timeout=heartbeat_timeout,
            )