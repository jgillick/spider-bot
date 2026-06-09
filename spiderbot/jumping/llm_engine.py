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
# Reward function inventory (verified at runtime from genesis_forge.mdp.rewards)
# ---------------------------------------------------------------------------

_PLAIN_FUNCTIONS = [
    "action_rate_l2",
    "ang_vel_xy_l2",
    "base_height",
    "command_tracking_ang_vel",
    "command_tracking_lin_vel",
    "contact_force",
    "dof_similar_to_default",
    "dof_torque_l2",
    "dof_velocity_l2",
    "entity_ang_vel",
    "entity_lin_vel",
    "entity_projected_gravity",
    "feet_air_time",
    "feet_ground_time",
    "feet_slide",
    "flat_orientation_l2",
    "has_contact",
    "is_alive",
    "lin_vel_xy_l2",
    "lin_vel_z_l2",
    "stand_still_joint_deviation_l1",
    "terminated",
]

# These are MdpFnClass instances — referenced in the cfg dict as the CLASS,
# not as a callable.  e.g. "fn": rewards.action_acceleration_l2 (not called).
_MDPFNCLASS_ENTRIES = [
    "action_acceleration_l2",
    "body_acceleration_exp",
]

_EXCLUDED_TERMS = [
    "flat_orientation",       # actively opposes jumping — penalises body rotation
    "flat_orientation_l2",    # same
    "base_height",            # targeting resting height prevents lifting off
    "has_contact",            # for stable footing use — penalises all-feet-airborne
    "command_tracking_lin_vel",  # locomotion gait term irrelevant to jumping
    "command_tracking_ang_vel",  # same
    "feet_ground_time",       # rewards keeping feet on ground — opposes jumping
]

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

## Available reward functions (exact API names)
Plain functions — used as `"fn": rewards.<name>`:
{plain_fn_list}

MdpFnClass entries — referenced as the CLASS, not called: use `"fn": rewards.<name>` (no parentheses):
{mdpfn_list}

## Excluded reward terms (MUST NOT use — they actively oppose jumping)
{excluded_list}

Rationale for exclusions:
- `flat_orientation*` penalises body tilt, which prevents the crouch-and-extend needed to jump
- `base_height` targeting resting height prevents the robot from lifting the body off the ground
- `has_contact` for stable footing rewards keeping feet planted
- locomotion gait terms (command_tracking_*) drive forward walking, not explosive jumping
- `feet_ground_time` rewards feet staying on the ground

## Code constraints
- Only import from `genesis_forge`, `spiderbot`, and stdlib
- Maintain the `config()` method signature exactly
- Do NOT modify files outside `spiderbot/jumping/`
- Do NOT add `height_command_manager` to ObservationManager — the jumping env has no velocity command manager
- Preserve the `body_terrain_contact` ContactManager (required for eval — do not remove it)
- Preserve `self_contact` ContactManager in config (used as a reward penalty source — keep the contact_force reward entry)
- Do NOT add a `self_contact` entry to TerminationManager
- The `dof_velocity_limit` termination (250 RPM threshold) must always remain present

## MdpFnClass usage example
```python
# WRONG — calling it like a function:
"fn": rewards.action_acceleration_l2,   # wrong if action_acceleration_l2 is MdpFnClass
# CORRECT — reference the class directly (no parentheses):
"fn": rewards.action_acceleration_l2,   # correct — MdpFnClass entries work this way
```
The distinction is that MdpFnClass entries are self-contained objects that genesis_forge
calls with the env as the argument; you reference them as classes, not instances.

## Response format
Return ONLY file content blocks. Use this exact delimiter format:

--- environment.py ---
<complete Python file content>
--- end ---

You may also propose changes to `ppo.yaml` using the same format.
ALWAYS include a `--- reasoning ---` / `--- end ---` block explaining what you changed and why.
The reasoning block must come LAST.
""".format(
    plain_fn_list="\n".join(f"  - {f}" for f in _PLAIN_FUNCTIONS),
    mdpfn_list="\n".join(f"  - {f}" for f in _MDPFNCLASS_ENTRIES),
    excluded_list="\n".join(f"  - {f}" for f in _EXCLUDED_TERMS),
)

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
    else:
        lines.append("\n## Run history\nNo prior runs yet — this is iteration 1.\n")

    lines.append(
        "\nRemember: use the exact delimiter format `--- <filename> ---` / `--- end ---` "
        "for each file you want to modify. End with a `--- reasoning ---` block.\n"
        "Only include files you actually want to change."
    )

    return "\n".join(lines)
