"""
LLM reasoning engine for the autonomous jumping agent.
Calls Claude to propose modifications to the jumping environment files,
validates the generated Python, and returns structured results.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Literal

import anthropic

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ProposalResult:
    modified_files: dict[str, str]
    reasoning: str
    mode: str


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an RL reward engineer for a spider robot with 8 legs (4 on each side).
Your task is to design reward functions that teach the robot to JUMP FORWARD.

## Target behavior
The robot must:
1. Get all 8 feet simultaneously off the ground for at least 1 step (true jump)
2. Achieve as much vertical height and forward distance as possible
3. Land with ONLY the feet making contact with the ground
4. Non-foot body parts (femur, tibia, motor housings, body) must not strike the ground hard at landing

## Success criteria (hard thresholds)
- CoM height must exceed resting height by at least 0.02 m during flight
- Forward distance (x-axis) must be positive
- Peak contact force on non-feet body links must stay below 15 N at landing

## Available managers in self (ONLY use what is listed here)
These are the attributes you can reference in params:
  - self.actuator_manager   — ActuatorManager (joint torques, positions, velocities)
  - self.action_manager     — PositionActionManager (raw actions, DOF positions/velocities)
  - self.robot_manager      — EntityManager (CoM position, linear/angular velocity)
  - self.foot_contact_manager — ContactManager, track_air_time=True (foot-to-terrain contacts)
  - self.self_contact        — ContactManager (robot-to-robot self contacts)
  - self.body_terrain_contact — ContactManager (non-foot body-to-terrain contacts; DO NOT REMOVE)

## Reward function reference (genesis_forge.mdp.rewards)
These are ALL available functions. Use ONLY these exact names with ONLY the listed params.
DO NOT invent parameters — any unknown keyword argument will crash at runtime.

### No required params (just use `"fn": rewards.<name>`)
  - is_alive(env)                     → 1.0 if robot alive
  - terminated(env)                   → 1.0 on termination
  - action_rate_l2(env)               → L2 penalty on action change rate

### Required: actuator_manager
  - dof_torque_l2(env, actuator_manager)
    params: {"actuator_manager": self.actuator_manager}

### Required: action_manager
  - dof_velocity_l2(env, action_manager)
    params: {"action_manager": self.action_manager}

### Required: actuator_manager OR action_manager (at least one)
  - dof_similar_to_default(env, actuator_manager=None, action_manager=None)
    params: {"actuator_manager": self.actuator_manager}

### Required: contact_manager; optional threshold (default 1.0)
  - contact_force(env, contact_manager, threshold=1.0)
    params: {"contact_manager": self.self_contact}
  - has_contact(env, contact_manager, threshold=1.0, min_contacts=1)
    EXCLUDED for jumping (see below)

### Required: contact_manager + time_threshold; contact_manager MUST have track_air_time=True
  - feet_air_time(env, contact_manager, time_threshold, time_threshold_max=None)
    params: {"contact_manager": self.foot_contact_manager, "time_threshold": 0.2}
    NOTE: self.foot_contact_manager already has track_air_time=True. Do NOT use other contact managers here.

### Optional: entity_attr (default "robot") OR entity_manager
  - lin_vel_z_l2(env, entity_attr="robot", entity_manager=None)
    params: {}   ← works with no params (defaults to robot)
  - ang_vel_xy_l2(env, entity_attr="robot", entity_manager=None)
    params: {}   ← works with no params
  - flat_orientation_l2(env, entity_attr="robot", entity_manager=None)
    EXCLUDED for jumping (see below)

### Required: entity_manager (not entity_attr — this one has NO default)
  - lin_vel_xy_l2(env, entity_manager)
    params: {"entity_manager": self.robot_manager}

### MdpFnClass — reference the CLASS directly as `"fn": rewards.<name>` (no params needed)
  - action_acceleration_l2   — penalises jittery action oscillations
    params: {}   ← accepts no params
  - body_acceleration_exp(env, entity_attr="robot", entity_manager=None, sensitivity=0.10)
    params: {}   ← works with no params (defaults to robot, sensitivity=0.10)
    Optional override: {"sensitivity": 0.05}  ← ONLY valid param besides entity_attr/entity_manager

## INTERNAL UTILITIES — NOT reward functions (DO NOT USE in reward config)
entity_lin_vel, entity_ang_vel, entity_projected_gravity are internal helpers.
They take a RigidEntity, not an env. Do NOT put them in the RewardManager config.

## Excluded reward terms (MUST NOT use — they actively oppose jumping)
  - flat_orientation_l2   — penalises body tilt needed for jump takeoff
  - base_height           — targeting resting height prevents lifting off
  - has_contact           — rewards feet staying planted on the ground
  - command_tracking_lin_vel / command_tracking_ang_vel — locomotion gait terms
  - feet_ground_time      — penalises feet leaving the ground

## Reward config dict format (CRITICAL — KeyError crashes if wrong)
Every reward entry MUST have a `"fn"` key and a `"weight"` key:
```python
self.reward_manager = RewardManager(
    self,
    cfg={
        "my_reward_name": {
            "weight": -0.01,           # REQUIRED: float (negative = penalty)
            "fn": rewards.action_rate_l2,  # REQUIRED: the function/class reference
            "params": {},              # OPTIONAL: omit entirely if no params needed
        },
        "another_reward": {
            "weight": 1.0,
            "fn": rewards.dof_torque_l2,
            "params": {"actuator_manager": self.actuator_manager},
        },
    },
)
```
DO NOT use any key besides `"weight"`, `"fn"`, and `"params"`.

## Code constraints
- Only import from `genesis_forge`, `spiderbot`, and stdlib
- Maintain the `config()` method signature exactly
- Do NOT add `height_command_manager` to ObservationManager (jumping env has none)
- Preserve `body_terrain_contact` ContactManager (required for eval)
- Preserve `self_contact` ContactManager (monitoring + reward source)
- Do NOT add `self_contact` to TerminationManager
- The `dof_velocity_limit` termination (250 RPM) must always remain present

## Response format
Return ONLY file content blocks in this exact format:

--- environment.py ---
<complete Python file content>
--- end ---

You may also include `--- ppo.yaml ---` blocks.
ALWAYS end with a `--- reasoning ---` / `--- end ---` block.
"""

# ---------------------------------------------------------------------------
# File block parsing
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"---\s*(?P<name>[\w./]+)\s*---\n(?P<content>.*?)(?=---\s*[\w./]+\s*---|$)",
    re.DOTALL,
)
_END_MARKER_RE = re.compile(r"---\s*end\s*---", re.IGNORECASE)


def _parse_blocks(text: str) -> dict[str, str]:
    """
    Extract `--- <name> ---` ... `--- end ---` blocks from the LLM response.
    Returns a dict mapping block name to raw content (stripped).
    """
    blocks: dict[str, str] = {}
    parts = re.split(r"---\s*([\w./]+)\s*---", text)
    # parts alternates: [pre, name0, content0, name1, content1, ...]
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        raw = parts[i + 1]
        # Strip a trailing `--- end ---` if present
        raw = _END_MARKER_RE.split(raw)[0]
        blocks[name] = raw.strip()
    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_modifications(
    current_files: dict[str, str],
    run_history: list[dict],
    prompt_mode: Literal["explore", "tune"],
    model: str = "claude-sonnet-4-6",
) -> ProposalResult | None:
    """
    Call Claude to propose modifications to the jumping environment.

    Args:
        current_files: dict mapping filename → content for the files Claude may modify.
        run_history: Full run history list; only the last 5 entries are sent.
        prompt_mode: "explore" → qualitatively new reward structure;
                     "tune" → adjust weights only within the current structure.
        model: Anthropic model ID.

    Returns:
        ProposalResult with modified_files and reasoning, or None if all retries fail.
    """
    user_message = _build_user_message(current_files, run_history, prompt_mode)
    client = anthropic.Anthropic()

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError:
            raise

        text = response.content[0].text
        blocks = _parse_blocks(text)

        reasoning = blocks.pop("reasoning", "")
        if not reasoning:
            # Extract any text before the first file block as fallback reasoning
            first_block_pos = text.find("---")
            if first_block_pos > 0:
                reasoning = text[:first_block_pos].strip()

        # Validate Python files; accumulate errors for retry
        errors: list[str] = []
        for filename, content in blocks.items():
            if not filename.endswith(".py"):
                continue
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{filename}: {e}")

        if not errors:
            return ProposalResult(
                modified_files=blocks,
                reasoning=reasoning,
                mode=prompt_mode,
            )

        if attempt == 0:
            # Append errors to the user message and retry once
            error_text = "\n".join(errors)
            user_message = (
                user_message
                + f"\n\nYour previous response contained syntax errors. "
                f"Fix them and try again:\n{error_text}"
            )

    return None


def _build_user_message(
    current_files: dict[str, str],
    run_history: list[dict],
    prompt_mode: Literal["explore", "tune"],
) -> str:
    lines: list[str] = []

    if prompt_mode == "explore":
        lines.append(
            "## Task: EXPLORE — propose a qualitatively different reward structure\n"
            "Change reward terms, add new terms, or change which behaviors are rewarded/penalised. "
            "Do not simply adjust weights of the existing terms — propose a different hypothesis "
            "about what will cause the robot to jump."
        )
    else:
        lines.append(
            "## Task: TUNE — adjust reward weights only\n"
            "The current reward structure showed some promise. Adjust only the numeric weight values "
            "to improve jump quality. Preserve all reward term names exactly as-is."
        )

    lines.append("\n## Current environment files\n")
    for filename, content in current_files.items():
        lines.append(f"### {filename}\n```python\n{content}\n```\n")

    recent = run_history[-5:] if len(run_history) > 5 else run_history
    if recent:
        lines.append("\n## Recent run history (last 5 runs)\n")
        lines.append("```json\n" + json.dumps(recent, indent=2) + "\n```\n")

        # Surface any subprocess errors prominently so the LLM can fix them
        error_entries = [
            e for e in recent
            if e.get("stop_reason") == "error" and e.get("metrics", {}).get("error_tail")
        ]
        if error_entries:
            lines.append("\n## Subprocess errors from recent runs\n")
            lines.append(
                "The following runs crashed at runtime. Fix these errors in your new proposal:\n"
            )
            for entry in error_entries[-3:]:
                tail = entry["metrics"]["error_tail"]
                lines.append(
                    f"Iteration {entry['iteration']} ({entry['run_type']}):\n"
                    f"```\n{tail}\n```\n"
                )
    else:
        lines.append("\n## Run history\nNo prior runs yet — this is iteration 1.\n")

    lines.append(
        "\nRemember: use the exact delimiter format `--- <filename> ---` / `--- end ---` "
        "for each file you want to modify. End with a `--- reasoning ---` block.\n"
        "Only include files you actually want to change."
    )

    return "\n".join(lines)
