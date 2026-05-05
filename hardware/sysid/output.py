"""
Terminal output formatting for sysid results.

Pure formatting — no hardware calls, no fitting logic.

Parameter → sim mapping:
  J         → armature     MuJoCo <joint armature=>,   URDF <dynamics armature=>,  genesis-forge armature
  b         → damping      MuJoCo <joint damping=>,    URDF <dynamics damping=>,   genesis-forge damping
  τ_c       → frictionloss MuJoCo <joint frictionloss=>, URDF <dynamics friction=>, genesis-forge frictionloss
  kp        → position kp  MuJoCo <position kp=>,      (no URDF equivalent),       genesis-forge kp
  kd        → position kv  MuJoCo <position kv=>,      (no URDF equivalent),       genesis-forge kv

Output order per result: MuJoCo → URDF → genesis-forge
"""

from __future__ import annotations

_SECTION_WIDTH = 60


def _header(title: str) -> str:
    return f"\n{'─' * _SECTION_WIDTH}\n{title}\n{'─' * _SECTION_WIDTH}"


def _noisy_value(nominal: float, half_range: float) -> str:
    return f"NoisyValue({nominal:.4f}, {half_range:.4f})"


def _dr_range(nominal: float, spread: float, floor_pct: float = 0.0) -> tuple[float, float]:
    half = max(spread / 2.0, nominal * floor_pct)
    return nominal - half, nominal + half


def format_inertia_result(
    J: float,
    b: float,
    J_spread: float,
    b_spread: float,
) -> str:
    """Format rotor inertia and viscous damping results."""
    J_lo, J_hi = _dr_range(J, J_spread)
    b_lo, b_hi = _dr_range(b, b_spread)

    lines = [_header("Rotor Inertia (J) + Viscous Damping (b)")]
    lines.append(f"  J:  {J:.6f} kg·m²     (spread: {J_spread:.6f})  → DR range: ({J_lo:.6f}, {J_hi:.6f})")
    lines.append(f"  b:  {b:.6f} N·m·s/rad  (spread: {b_spread:.6f})  → DR range: ({b_lo:.6f}, {b_hi:.6f})")
    lines.append("")
    lines.append("  MuJoCo MJCF:")
    lines.append(f'    <joint ... armature="{J:.6f}" damping="{b:.6f}"/>')
    lines.append("")
    lines.append("  URDF <dynamics>:")
    lines.append(f'    <dynamics armature="{J:.6f}" damping="{b:.6f}"/>')
    lines.append("")
    lines.append("  genesis-forge ActuatorManager:")
    lines.append(f"    armature={_noisy_value(J, J_spread / 2)}")
    lines.append(f"    damping={_noisy_value(b, b_spread / 2)}")
    return "\n".join(lines)


def format_coulomb_result(
    tau_c_cw: float,
    tau_c_ccw: float,
    spread_cw: float,
    spread_ccw: float,
) -> str:
    """Format Coulomb friction results (per direction)."""
    # DR floor: ±50% of nominal (Coulomb is the biggest sim-to-real killer)
    floor_pct = 0.50
    cw_lo, cw_hi = _dr_range(tau_c_cw, spread_cw, floor_pct)
    ccw_lo, ccw_hi = _dr_range(tau_c_ccw, spread_ccw, floor_pct)
    tau_c_nominal = (tau_c_cw + tau_c_ccw) / 2.0
    half_range = max((spread_cw + spread_ccw) / 4.0, tau_c_nominal * floor_pct)

    lines = [_header("Coulomb Friction (τ_c)")]
    lines.append(f"  CW:  {tau_c_cw:.4f} N·m  (spread: {spread_cw:.4f})  → DR range: ({cw_lo:.4f}, {cw_hi:.4f})")
    lines.append(f"  CCW: {tau_c_ccw:.4f} N·m  (spread: {spread_ccw:.4f})  → DR range: ({ccw_lo:.4f}, {ccw_hi:.4f})")
    lines.append("  Note: a single scalar nominal is used; average of CW and CCW.")
    lines.append("")
    lines.append("  MuJoCo MJCF:")
    lines.append(f'    <joint ... frictionloss="{tau_c_nominal:.4f}"/>')
    lines.append("")
    lines.append("  URDF <dynamics>:")
    lines.append(f'    <dynamics friction="{tau_c_nominal:.4f}"/>')
    lines.append("")
    lines.append("  genesis-forge ActuatorManager:")
    lines.append(f"    frictionloss={_noisy_value(tau_c_nominal, half_range)}")
    return "\n".join(lines)


def format_static_result(
    tau_static_cw: float,
    tau_static_ccw: float,
) -> str:
    """Format static (breakaway) friction results."""
    lines = [_header("Static Friction (τ_static)")]
    lines.append(f"  CW:  {tau_static_cw:.4f} N·m")
    lines.append(f"  CCW: {tau_static_ccw:.4f} N·m")
    lines.append("")
    lines.append("  Static friction sets the minimum torque needed to break from rest.")
    lines.append("  It should be ≥ Coulomb friction — use it to sanity-check that result.")
    lines.append("  MuJoCo, URDF, and genesis-forge do not model static friction separately;")
    lines.append("  no copy-paste snippet is provided.")
    return "\n".join(lines)


def format_kp_kd_result(kp: float, kd: float) -> str:
    """Format kp/kd step-response results."""
    kp_half = kp * 0.15
    kd_half = kd * 0.15

    lines = [_header("Impedance Gains (kp / kd)")]
    lines.append(f"  kp: {kp:.4f} N·m/rad")
    lines.append(f"  kd: {kd:.4f} N·m·s/rad")
    lines.append("  Note: kd is labeled 'kv' in genesis-forge and MuJoCo position actuators.")
    lines.append("")
    lines.append("  MuJoCo MJCF:")
    lines.append(f'    <position joint="..." kp="{kp:.4f}" kv="{kd:.4f}"/>')
    lines.append("")
    lines.append("  URDF:")
    lines.append("    No direct equivalent — kp/kd are controller gains, not plant parameters.")
    lines.append("")
    lines.append("  genesis-forge ActuatorManager:")
    lines.append(f"    kp={_noisy_value(kp, kp_half)}")
    lines.append(f"    kv={_noisy_value(kd, kd_half)}")
    return "\n".join(lines)
