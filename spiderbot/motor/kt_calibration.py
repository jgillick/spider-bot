"""
Measure a leg motor's real torque constant (Kt) with a lever arm and a scale.

Rig: clamp the motor to the bench and bolt a stiff arm to its output. The arm's contact
point, a measured distance from the motor axis (--lever), presses down on a kitchen scale.
Tare the scale with the arm resting on it and the motor idle, so readings are only the
force the motor adds.

The script steps the motor through torque commands, ramping between them and slowly back
to zero after the last one so the arm doesn't jump off the scale. At each step it averages
the measured phase current (Iq), bus power and the firmware's own torque estimate, then
asks you for the scale reading. It then fits:

  - Kt:       real torque per amp of Iq (slope of τ against Iq)
  - k_w:      bus watts per (N·m)², the coefficient the sim's power model uses
  - firmware: how far the firmware's torque estimate is from the real torque, and the
              torque_constant it should have

Steps are in *commanded* torque, which the firmware converts to current with its own
torque_constant. If that constant is wrong, the real torque differs: with the firmware at
0.43 and a real Kt of 1.0, commanding 3 N·m gives about 7 N·m.

Safety: torque mode has no position hold, so if the arm slips off the scale the motor
spins up. A background thread watches speed and position and cuts the torque the moment
either one moves past its limit.

Run:
    python -m spiderbot.motor.kt_calibration --lever 0.20
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from spiderbot.motor.can_motor_controller import CanMotorController
from spiderbot.motor.protocol import AxisState, CANCommand, ControlMode, InputMode

GRAVITY = 9.80665  # m/s²

DEFAULT_STEPS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # commanded N·m
DEFAULT_MAX_TORQUE = 4.0  # refuse any commanded step above this (N·m)
DEFAULT_TORQUE_LIMIT = 18.0  # N·m, the sim's max_force; used for the suggested limits
CURRENT_LIMIT_MARGIN = 1.2  # current_lim headroom over what torque_lim needs
DEFAULT_RAMP_RATE = 2.0  # N·m/s between steps
DEFAULT_RELEASE_RATE = 0.75  # N·m/s back to zero after the last step

# Abort when the arm moves: on the scale it should barely turn.
MAX_SPEED = 1.0  # rad/s
MAX_TRAVEL = 0.15  # rad from where it started (about 9°)

CONTROL_HZ = 100
TELEMETRY_EVERY = 5  # request telemetry every Nth control loop (20 Hz)
TELEMETRY_COMMANDS = (
    CANCommand.GET_IQ,
    CANCommand.GET_BUS_VOLTAGE_CURRENT,
    CANCommand.GET_TORQUES,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")


# ---------------------------------------------------------------------------
# Analysis (pure functions)
# ---------------------------------------------------------------------------


@dataclass
class LineFit:
    slope: float
    intercept: float
    r2: float


def fit_line(x: list[float], y: list[float]) -> LineFit:
    """Least-squares fit of y = slope·x + intercept."""
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 points to fit a line")
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        raise ValueError("x values are all the same")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LineFit(slope, intercept, r2)


def step_toward(current: float, target: float, max_delta: float) -> float:
    """Move *current* toward *target* by at most *max_delta*."""
    if abs(target - current) <= max_delta:
        return target
    return current + math.copysign(max_delta, target - current)


@dataclass
class StepResult:
    commanded: float  # N·m
    iq: float  # A
    bus_voltage: float  # V
    bus_current: float  # A
    firmware_torque: float  # N·m, the firmware's own estimate
    scale_kg: float
    lever: float  # m

    @property
    def torque(self) -> float:
        """Real torque from the scale, N·m."""
        return self.scale_kg * GRAVITY * self.lever

    @property
    def bus_power(self) -> float:
        return self.bus_voltage * self.bus_current


@dataclass
class Calibration:
    kt: LineFit  # τ = kt·Iq + offset
    power: LineFit  # P = k_w·τ² + idle
    firmware_ratio: float  # real torque / firmware estimate
    firmware_kt: float  # what the firmware currently uses (its estimate / Iq)
    suggested_firmware_kt: float


def analyze(results: list[StepResult]) -> Calibration:
    iq = [abs(r.iq) for r in results]
    torque = [r.torque for r in results]
    kt = fit_line(iq, torque)
    power = fit_line([t * t for t in torque], [r.bus_power for r in results])
    fw = [abs(r.firmware_torque) for r in results]
    firmware_ratio = sum(torque) / sum(fw)
    firmware_kt = sum(fw) / sum(iq)
    return Calibration(
        kt=kt,
        power=power,
        firmware_ratio=firmware_ratio,
        firmware_kt=firmware_kt,
        suggested_firmware_kt=firmware_kt * firmware_ratio,
    )


def current_limit_for(torque_limit: float, kt: float, margin: float = CURRENT_LIMIT_MARGIN) -> float:
    """Phase current limit that lets the motor reach *torque_limit*, with some headroom."""
    return margin * torque_limit / kt


def format_report(
    results: list[StepResult],
    cal: Calibration,
    sim: dict[str, float] | None = None,
    torque_limit: float = DEFAULT_TORQUE_LIMIT,
) -> str:
    lines = [
        "",
        f"{'cmd N·m':>8}{'Iq A':>8}{'scale kg':>10}{'real N·m':>10}{'fw N·m':>8}"
        f"{'Vbus':>7}{'Ibus A':>8}{'P W':>7}",
    ]
    for r in results:
        lines.append(
            f"{r.commanded:>8.2f}{r.iq:>8.2f}{r.scale_kg:>10.3f}{r.torque:>10.2f}"
            f"{r.firmware_torque:>8.2f}{r.bus_voltage:>7.1f}{r.bus_current:>8.2f}{r.bus_power:>7.1f}"
        )
    lines += [
        "",
        f"Kt (real torque per amp):  {cal.kt.slope:.3f} N·m/A   "
        f"(offset {cal.kt.intercept:+.2f} N·m, r² {cal.kt.r2:.4f})",
        f"k_w (bus W per N·m²):      {cal.power.slope:.3f}   "
        f"(idle {cal.power.intercept:.1f} W, r² {cal.power.r2:.4f})",
        f"Firmware torque_constant:  {cal.firmware_kt:.3f} now -> {cal.suggested_firmware_kt:.3f} "
        f"(real torque is {cal.firmware_ratio:.2f}× what the firmware reports)",
    ]
    if cal.kt.r2 < 0.99:
        lines.append(
            "⚠️  The Kt fit is loose (r² < 0.99). Check the arm stays perpendicular, "
            "the scale was tared, and nothing rubs."
        )
    if abs(cal.kt.intercept) > 0.1 * max(r.torque for r in results):
        lines.append(
            "⚠️  Large torque offset. Usually a tare error, friction, or the arm's own weight."
        )

    kt = cal.kt.slope
    current_lim = current_limit_for(torque_limit, kt)
    lines += [
        "",
        "Set on the motor controller (odrivetool, then save_configuration()):",
        f"  axis0.motor.config.torque_constant = {cal.suggested_firmware_kt:.4f}"
        f"   # was {cal.firmware_kt:.4f}",
        f"  axis0.motor.config.torque_lim      = {torque_limit:.1f}"
        f"      # N·m, the sim's max_force",
        f"  axis0.motor.config.current_lim     = {current_lim:.1f}"
        f"      # A, {CURRENT_LIMIT_MARGIN:.1f}× what {torque_limit:.0f} N·m needs at Kt {kt:.3f}",
    ]
    if sim:
        lines += [
            "",
            "Set in the sim (LEG_MOTOR in spiderbot/locomotion/environment.py):",
            f"  torque_constant = {kt:.3f}   # was {sim['torque_constant']:.3f}",
        ]
        if "phase_resistance" in sim:
            predicted = 1.5 * sim["phase_resistance"] / kt**2
            lines.append(
                f"  Power model check: phase_resistance {sim['phase_resistance']:.3f} Ω with this Kt "
                f"predicts {predicted:.3f} W/N·m²; measured {cal.power.slope:.3f}."
            )
            if not 0.8 <= cal.power.slope / predicted <= 1.25:
                lines.append(
                    "  ⚠️  They differ by more than ~20%. The sim's power numbers will be off by "
                    "about that much; check phase_resistance (and measure it at room temperature)."
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


class TorqueHolder:
    """
    Background thread that streams a ramped torque setpoint to one motor and stops it
    if the arm moves. The main thread sets targets and can block on input() meanwhile.
    """

    def __init__(
        self,
        bus: CanMotorController,
        node_id: int,
        direction: float,
        ramp_rate: float,
        max_speed: float = MAX_SPEED,
        max_travel: float = MAX_TRAVEL,
    ):
        self.bus = bus
        self.node_id = node_id
        self.direction = direction
        self.ramp_rate = ramp_rate
        self.max_speed = max_speed
        self.max_travel = max_travel

        self.target = 0.0
        self.setpoint = 0.0
        self.fault: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_pos: float | None = None

    def start(self) -> None:
        self._start_pos = self.bus.get_state(self.node_id).position
        self._thread = threading.Thread(target=self._run, daemon=True, name="torque-holder")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_target(self, torque: float, ramp_rate: float | None = None) -> None:
        with self._lock:
            self.target = torque
            if ramp_rate is not None:
                self.ramp_rate = ramp_rate

    def wait_until_reached(self, timeout: float = 30.0) -> bool:
        """Block until the setpoint reaches the target. False on fault or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.fault:
                return False
            with self._lock:
                if self.setpoint == self.target:
                    return True
            time.sleep(0.01)
        return False

    def _trip(self, reason: str) -> None:
        self.fault = reason
        try:
            self.bus.send_torque(self.node_id, 0.0)
            self.bus.send_set_state(self.node_id, AxisState.IDLE)
        finally:
            self._stop.set()

    def _run(self) -> None:
        dt = 1.0 / CONTROL_HZ
        loop = 0
        while not self._stop.is_set():
            state = self.bus.get_state(self.node_id)
            if abs(state.velocity) > self.max_speed:
                self._trip(f"arm moving at {state.velocity:.2f} rad/s (slipped off the scale?)")
                return
            if abs(state.position - self._start_pos) > self.max_travel:
                travel = state.position - self._start_pos
                self._trip(f"arm turned {math.degrees(travel):.1f}° from its start")
                return
            if state.axis_error:
                self._trip(f"axis error 0x{state.axis_error:x}")
                return

            with self._lock:
                self.setpoint = step_toward(self.setpoint, self.target, self.ramp_rate * dt)
                setpoint = self.setpoint
            self.bus.send_torque(self.node_id, self.direction * setpoint)
            if loop % TELEMETRY_EVERY == 0:
                self.bus.request_telemetry(self.node_id, TELEMETRY_COMMANDS)
            loop += 1
            time.sleep(dt)


def sample_telemetry(
    bus: CanMotorController, node_id: int, duration: float, holder: TorqueHolder
) -> dict[str, float]:
    """Average fresh telemetry replies over *duration* seconds."""
    sums = {"iq": 0.0, "bus_voltage": 0.0, "bus_current": 0.0, "firmware_torque": 0.0}
    counts = dict.fromkeys(sums, 0)
    seen: dict[str, float | None] = {"iq": None, "bus": None, "torque": None}
    end = time.monotonic() + duration
    while time.monotonic() < end and not holder.fault:
        t = bus.get_telemetry(node_id)
        if t.iq_setpoint_time is not None and t.iq_setpoint_time != seen["iq"]:
            seen["iq"] = t.iq_setpoint_time
            sums["iq"] += t.iq_measured
            counts["iq"] += 1
        if t.bus_voltage_time is not None and t.bus_voltage_time != seen["bus"]:
            seen["bus"] = t.bus_voltage_time
            sums["bus_voltage"] += t.bus_voltage
            sums["bus_current"] += t.bus_current
            counts["bus_voltage"] += 1
            counts["bus_current"] += 1
        if t.torque_target_time is not None and t.torque_target_time != seen["torque"]:
            seen["torque"] = t.torque_target_time
            sums["firmware_torque"] += t.torque_estimate
            counts["firmware_torque"] += 1
        time.sleep(0.01)
    missing = [k for k, c in counts.items() if c == 0]
    if missing and not holder.fault:
        raise RuntimeError(
            f"No telemetry replies for {missing}. If the firmware ignores RTR requests, "
            "enable cyclic messages (axis0.config.can.iq_rate_ms, bus_vi_rate_ms = 50)."
        )
    return {k: sums[k] / counts[k] if counts[k] else float("nan") for k in sums}


def read_scale(prompt: str) -> float:
    """Ask for a scale reading in kg ("812g" also works)."""
    while True:
        raw = input(prompt).strip().lower().replace(" ", "")
        try:
            if raw.endswith("kg"):
                return abs(float(raw[:-2]))
            if raw.endswith("g"):
                return abs(float(raw[:-1])) / 1000.0
            return abs(float(raw))
        except ValueError:
            print("  Enter a number in kg, e.g. 1.234 or 1234g")


def enable_torque_mode(bus: CanMotorController, node_id: int) -> None:
    bus.send_clear_errors(node_id)
    time.sleep(0.05)
    bus.send_set_control_mode(node_id, ControlMode.TORQUE_CONTROL, InputMode.PASSTHROUGH)
    time.sleep(0.05)
    bus.send_torque(node_id, 0.0)
    time.sleep(0.05)
    bus.send_set_state(node_id, AxisState.CLOSED_LOOP_CONTROL)
    deadline = time.monotonic() + 2.0
    while bus.get_state(node_id).axis_state != AxisState.CLOSED_LOOP_CONTROL:
        if time.monotonic() > deadline:
            raise RuntimeError("Motor did not enter closed-loop control")
        time.sleep(0.02)
    # The cyclic encoder estimates read 0 while idle; wait for real ones before the
    # torque holder records its starting position.
    time.sleep(0.3)


def connect(node_id: int) -> CanMotorController:
    from spiderbot.locomotion.leg_motors import CAN_BITRATE, choose_port

    port = choose_port()
    bus = CanMotorController(channel=port, bitrate=CAN_BITRATE)
    bus.register_node(node_id)
    bus.start()
    deadline = time.monotonic() + 5.0
    while not bus.get_state(node_id).is_connected:
        if time.monotonic() > deadline:
            bus.stop()
            raise RuntimeError(f"No heartbeat from node {node_id}")
        time.sleep(0.05)
    time.sleep(0.2)  # let encoder estimates arrive
    return bus


def write_csv(path: str, results: list[StepResult]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commanded_nm", "iq_a", "scale_kg", "lever_m", "torque_nm",
                    "firmware_torque_nm", "bus_voltage", "bus_current", "bus_power_w"])
        for r in results:
            w.writerow([r.commanded, r.iq, r.scale_kg, r.lever, r.torque,
                        r.firmware_torque, r.bus_voltage, r.bus_current, r.bus_power])


def sim_motor_spec() -> dict[str, float] | None:
    """Read LEG_MOTOR's constants from the locomotion env without importing Genesis."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "locomotion", "environment.py")
    try:
        with open(path) as f:
            block = re.search(r"LEG_MOTOR\s*=\s*MotorSpec\((.*?)\)", f.read(), re.S)
    except OSError:
        return None
    if not block:
        return None
    spec = {
        name: float(value)
        for name, value in re.findall(r"(\w+)\s*=\s*([0-9.]+(?:[eE][-+]?\d+)?)", block.group(1))
    }
    return spec if "torque_constant" in spec else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lever", type=float, required=True,
                   help="distance from the motor axis to the scale contact point, in meters")
    p.add_argument("--node", type=int, default=1, help="CAN node ID (default 1)")
    p.add_argument("--steps", default=",".join(str(s) for s in DEFAULT_STEPS),
                   help="commanded torque steps in N·m (default %(default)s)")
    p.add_argument("--direction", type=int, choices=(1, -1), default=1,
                   help="torque sign that presses the arm into the scale (default 1)")
    p.add_argument("--max-torque", type=float, default=DEFAULT_MAX_TORQUE,
                   help="refuse steps above this commanded torque (default %(default)s)")
    p.add_argument("--torque-limit", type=float, default=DEFAULT_TORQUE_LIMIT,
                   help="output torque limit to suggest for the controller (default %(default)s N·m)")
    p.add_argument("--ramp-rate", type=float, default=DEFAULT_RAMP_RATE,
                   help="N·m/s between steps (default %(default)s)")
    p.add_argument("--release-rate", type=float, default=DEFAULT_RELEASE_RATE,
                   help="N·m/s back to zero after the last step (default %(default)s)")
    p.add_argument("--settle", type=float, default=1.0, help="seconds to settle at each step")
    p.add_argument("--sample", type=float, default=2.0, help="seconds of telemetry to average")
    p.add_argument("--out", help="CSV path (default logs/kt_calibration_<timestamp>.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    steps = [float(s) for s in args.steps.split(",") if s.strip()]
    if not steps or any(s <= 0 for s in steps):
        raise SystemExit("--steps must be positive torques, e.g. 0.5,1,1.5")
    if max(steps) > args.max_torque:
        raise SystemExit(f"Step {max(steps)} N·m is above --max-torque {args.max_torque}")
    if args.lever <= 0:
        raise SystemExit("--lever must be positive (meters)")

    print(f"Lever {args.lever:.3f} m. Steps (commanded): {steps} N·m.")
    input("\n⚖️  Arm resting on the scale, motor idle, scale tared? Press Enter to start...")

    bus = connect(args.node)
    holder = TorqueHolder(bus, args.node, args.direction, args.ramp_rate)
    results: list[StepResult] = []
    try:
        enable_torque_mode(bus, args.node)
        holder.start()

        for i, commanded in enumerate(steps):
            holder.set_target(commanded, args.ramp_rate)
            if not holder.wait_until_reached():
                break
            time.sleep(args.settle)
            if holder.fault:
                break

            if i == 0:
                ok = input("Is the arm pressing harder on the scale (reading went up)? [y/N] ")
                if ok.strip().lower() != "y":
                    print("Releasing. Rerun with --direction", -args.direction)
                    break

            avg = sample_telemetry(bus, args.node, args.sample, holder)
            if holder.fault:
                break
            scale = read_scale(f"  {commanded:.2f} N·m commanded, Iq {avg['iq']:.2f} A. Scale reading (kg): ")
            if holder.fault:
                break
            results.append(StepResult(commanded=commanded, scale_kg=scale, lever=args.lever, **avg))

        # Ease off the scale slowly so the arm doesn't spring off it
        if not holder.fault:
            print("Releasing...")
            holder.set_target(0.0, args.release_rate)
            holder.wait_until_reached(timeout=max(steps) / args.release_rate + 5)
    except KeyboardInterrupt:
        print("\nInterrupted, releasing...")
        holder.set_target(0.0, args.release_rate * 2)
        holder.wait_until_reached(timeout=5)
    finally:
        holder.stop()
        bus.send_torque(args.node, 0.0)
        bus.send_set_state(args.node, AxisState.IDLE)
        time.sleep(0.1)
        bus.stop()

    if holder.fault:
        print(f"\n🛑 Stopped: {holder.fault}. Motor set to idle.")
    if len(results) < 2:
        print("Not enough steps to fit.")
        return

    out = args.out or os.path.join(
        LOG_DIR, f"kt_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    write_csv(out, results)
    print(format_report(results, analyze(results), sim_motor_spec(), args.torque_limit))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
