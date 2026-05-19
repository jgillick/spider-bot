# Motor Sysid

A CLI for identifying real motor parameters from a connected ODrive-compatible controller. Run these experiments on the bench, then copy-paste the results into `spider-bot/environment.py` and `model/SpiderBot.urdf` to replace the placeholder friction, damping, and gain values.

## Why

The sim-to-real pipeline uses guessed values for Coulomb friction, viscous damping, and impedance gains. Underestimated Coulomb friction is the most common cause of poor sim-to-real transfer for this class of motor. These scripts give you measured nominal values and principled domain randomization ranges.

## Prerequisites

Dependencies are already declared in `pyproject.toml`. From the repo root:

```
uv sync
```

Connect the motor controller over USB. The motor must be free to rotate through at least ±1 revolution during experiments.

## Before You Start

**Warm up the motor.** Run it for 5–10 minutes at operating temperature before collecting data. Cold measurements will be inaccurate.

## Verify Encoder Frame of Reference

Run this once per motor/controller combination to confirm the encoder reads rotor-side turns (not output-shaft turns). This affects all velocity-dependent parameters.

Start the sysid CLI and select menu option 1:

```
python -m hardware.sysid --gear-ratio 8
```

Then choose option 1 from the menu. Follow the prompt: rotate the output shaft exactly one full revolution, press Enter. Expected result for GIM8108-8:

```
→ ROTOR-SIDE position: encoder reads 8.00 turns per output shaft revolution.
  sysid fitting MUST divide vel_estimate by gear_ratio before fitting.
```

The fitting code already applies this conversion. This step confirms the assumption is correct before trusting the results.

## Run Experiments

```
python -m hardware.sysid --gear-ratio 8
```

The script connects over USB, initializes torque control mode autonomously, then shows a menu:

```
Sysid Menu:
  1. Verify encoder frame of reference  ← run first
  2. Rotor inertia (J) + Viscous damping (b)
  3. Coulomb friction (τ_c)
  4. Static friction (τ_static)
  5. Quit
```

**Run in order 1 → 2 → 3 → 4.** Run the encoder check once per motor/controller combination; it is not needed on every session thereafter.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--gear-ratio` | required | Rotor turns per output shaft turn (8 for GIM8108-8) |
| `--max-torque` | 1.0 N·m | Cap on all torque commands (motor-side). Raise if the motor stalls before motion onset during Coulomb/static experiments |
| `--trials` | 5 | Trials per experiment per direction. More trials → tighter DR ranges |
| `--kt` | from firmware | Override the torque constant Kt (N·m/A) read from firmware |
| `--serial` | first found | Target a specific controller when multiple are connected |
| `--timeout` | 10 s | USB connection timeout |
| `--poll-interval` | 0.01 s | Encoder polling rate (100 Hz) |

### Save Results

The script prints to stdout only. Capture everything with `tee`:

```
python -m hardware.sysid --gear-ratio 8 | tee sysid-results.txt
```

## Output

After each experiment the terminal prints fitted values, trial spread, and two copy-paste blocks:

```
────────────────────────────────────────────────────────────
Coulomb Friction (τ_c)
────────────────────────────────────────────────────────────
  CW:  0.142 N·m  (spread: 0.022)  → DR range: (0.071, 0.213)
  CCW: 0.119 N·m  (spread: 0.023)  → DR range: (0.060, 0.179)

  Maps to: frictionloss in genesis-forge, friction= in URDF <dynamics>

  genesis-forge ActuatorManager:
    frictionloss=NoisyValue(0.1305, 0.0653)

  URDF <dynamics>:
    <dynamics friction="0.1305" />
```

### Parameter Mapping

| Experiment | Measured | genesis-forge param | URDF attribute |
|---|---|---|---|
| 2 | Rotor inertia J | `armature` | `armature=` |
| 2 | Viscous damping b | `damping` | `damping=` |
| 3 | Coulomb friction τ_c | `frictionloss` | `friction=` |
| 4 | Static friction τ_static | (reference only) | — |

Paste the genesis-forge snippets into `spider-bot/environment.py` and the URDF snippets into `model/SpiderBot.urdf`.

## Notes

**CAN latency (`delay_steps`) cannot be measured over USB.** This parameter must be measured separately with a CAN adapter, or estimated: at 500 kbps with a 10 ms encoder broadcast period, `delay_steps = 1` at 50 Hz sim is a reasonable starting point.

**All torque values are motor-side** (before the gearbox). This is the correct reference frame for `armature` in genesis-forge.

**Ctrl-C is handled safely.** The script zeroes `input_torque` and idles the motor before exiting.
