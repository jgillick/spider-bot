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
Your task is to design reward functions that teach the robot to JUMP FORWARD as far as possible.

## Target behavior
The robot must, in priority order:
1. Maximize horizontal forward distance (x-axis displacement) on each jump — this is the PRIMARY objective
2. Get all 8 feet simultaneously off the ground for at least 1 step (true jump)
3. Land with ONLY the feet making contact with the ground
4. Non-foot body parts (femur, tibia, motor housings, body) must not strike the ground hard at landing

Vertical height is useful only insofar as it enables more forward distance — do NOT reward height for its own sake.

## Success criteria (hard thresholds, evaluated by the eval harness)
- Forward distance (x-axis) must exceed 0.3 m per jump
- CoM height must exceed resting height by at least 0.02 m (minimum jump confirmation — feet must leave the ground; this is not a height goal)
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
These are the built-in functions. Use ONLY these exact names with ONLY the listed params.
DO NOT invent parameters — any unknown keyword argument will crash at runtime.
For behaviors not expressible with these, write a custom function in `rewards.py` (see below).

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
  - action_acceleration_l2(env, action_manager=None)   — penalises jittery action oscillations
    params: {}   ← action_manager is optional; works with no params
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

## Custom reward functions (rewards.py)
When no genesis_forge function expresses the behavior you want, define it in `rewards.py`.

Signature:
  def fn(env) -> torch.Tensor   # (num_envs,) float — positive=reward, negative=penalty

Available data inside a custom reward:
  env.robot_manager.entity.get_pos()              → (num_envs, 3)   CoM xyz
  env.robot_manager.entity.get_pos()[:, 0]        → (num_envs,)     CoM x-position (PRIMARY: forward distance)
  env.robot_manager.entity.get_pos()[:, 2]        → (num_envs,)     CoM z-position (height)
  env.robot_manager.entity.get_vel()              → (num_envs, 3)   CoM linear velocity xyz
  env.robot_manager.entity.get_vel()[:, 0]        → (num_envs,)     CoM forward velocity (x)
  env.foot_contact_manager.contacts               → (num_envs, 8, 3) foot contact force vectors
  env.foot_contact_manager.contacts.norm(dim=-1)  → (num_envs, 8)   per-foot force magnitude
  env.body_terrain_contact.contacts               → contact forces for non-foot body links
  env.action_manager.get_dofs_position()          → (num_envs, num_dofs)
  env.action_manager.get_dofs_velocity()          → (num_envs, num_dofs)
Use torch operations only (no numpy).

Key design principle: reward COMPLETED forward displacement (Δx at landing), not instantaneous forward
velocity during flight. Rewarding vx densely during flight caused a 6.9m gallop with 2770 N slams in a
prior run — the robot found it could exploit forward speed without ever needing a clean landing.

Import and use in environment.py:
```python
from .rewards import my_fn              # relative import — required for experiment directory layout
...
"my_reward": {"weight": 1.0, "fn": my_fn}  # no "params" key needed
```
⚠️ Use RELATIVE imports (`from .rewards import fn_name`), NOT absolute ones.
   `from spiderbot.jumping import rewards` shadows genesis_forge.mdp.rewards.
   `from spiderbot.jumping.rewards import fn_name` imports from the WRONG file (base template, not this experiment).

## Observations — you MAY modify
The `ObservationManager` in `config()` controls what the policy sees.
You may add, remove, or reorder observation terms. Use `lambda env: ...` functions
that access `self` (the env) to reach any manager attribute.

Useful data sources for jumping:
  - Body height:        `lambda env: env.robot_manager.entity.get_pos()[:, 2:3]`
  - Body lin velocity:  `lambda env: env.robot_manager.entity.get_vel()[:, :3]`  (x/y/z)
  - Body z velocity:    `lambda env: env.robot_manager.entity.get_vel()[:, 2:3]`
  - Foot contact bool:  `lambda env: (env.foot_contact_manager.contacts.norm(dim=-1) > 1.0).float()`
  - DOF positions:      `lambda env: env.action_manager.get_dofs_position()`    ← already present
  - DOF velocities:     `lambda env: env.action_manager.get_dofs_velocity()`    ← already present
  - IMU:                `env.imu_observation`                                   ← already present
  - Raw actions:        `lambda env: env.action_manager.raw_actions`            ← already present

You may also add `noise` and `scale` keys to any observation entry:
  `"scale": 0.1` multiplies the observation tensor;  `"noise": 0.01` adds Gaussian noise.

DO NOT add `height_command_manager` — jumping env has none.

## Terminations — you MAY modify
The `TerminationManager` in `config()` controls when episodes reset.

Currently active:
  - `timeout`     — time_out=True; KEEP THIS (prevents infinite episodes)
  - `foot_angle`  — terminates if foot angle < -0.75 rad; MAY remove or retune
    ⚠️ This termination may prevent the robot from crouching before a jump.
    Consider removing it or raising the threshold magnitude (e.g. -1.5) to allow crouching.

Available termination functions (from `genesis_forge.mdp.terminations`):
  - `terminations.timeout(env)` — always needed
  - `terminations.contact_force(env, contact_manager, threshold=1.0)` — terminate if ANY link exceeds force threshold
    Example (body impact): `{"fn": terminations.contact_force, "params": {"contact_manager": self.body_terrain_contact, "threshold": 50.0}}`
  - `terminations.has_contact(env, contact_manager, threshold=1.0, min_contacts=1)` — terminate if min_contacts links are in contact
  - `terminations.contact_force_with_grace_period(env, contact_manager, threshold=100.0, grace_steps=10)` — contact_force with grace period at episode start
  - `terminations.bad_orientation(env, limit_angle=40.0, entity_attr="robot", grace_steps=0)` — terminate on excessive tilt
  - `terminations.is_upsidedown(env, threshold=0.5, entity_attr="robot", grace_steps=0)` — terminate if robot is inverted
  - `terminations.base_height_below_minimum(env, minimum_height=0.05, entity_attr="robot")` — terminate if CoM drops too low
  - `self.foot_angle_mdp.terminate(env, angle_threshold)` — custom foot-angle check

For each entry, `"time_out": True` marks it as a timeout (affects episode statistics).
Leave `time_out` out for non-timeout terminations.

## Custom termination functions (terminations.py)
When you need a termination condition not covered by the built-in functions, define it in `terminations.py`.

Signature:
  def fn(env) -> torch.Tensor   # (num_envs,) bool — True = reset this env

Import and use in environment.py:
```python
from .terminations import my_term_fn   # relative import — required
...
"my_term": {"fn": my_term_fn}
# add "time_out": True only if this is a timeout-type reset
```
⚠️ Use RELATIVE imports (`from .terminations import fn_name`), NOT absolute ones (same reason as rewards).

## Logging metrics to TensorBoard
RSL-RL reads `extras["episode"]` at episode boundaries and logs everything there to TensorBoard.
ALWAYS write to `extras["episode"]` — writing to the top-level `extras` dict will NOT be logged.

### Primary: log from `step()` in environment.py (preferred for success metrics)

`SpiderRobotJumpingEnv` already has a `step()` method you may extend. This is the RIGHT place
for episodic success metrics (jump distance, flight time, landing quality) because you can
accumulate state across steps independently of the reward structure.

```python
def step(self, actions: torch.Tensor):
    obs, reward, terminated, truncated, extras = super().step(actions)
    ep = extras["episode"]  # always write here, never to top-level extras

    # Per-step scalar metrics (mean across envs)
    x_vel = self.robot_manager.entity.get_vel()[:, 0]
    ep["Metrics / mean_forward_vel"] = x_vel.mean().clone()

    # Episodic metrics: accumulate state, reset on episode end
    if not hasattr(self, "_step_state"):
        self._step_state = {
            "max_x": torch.zeros(self.num_envs, device=gs.device),
            "flight_steps": torch.zeros(self.num_envs, dtype=torch.int32, device=gs.device),
        }
    s = self._step_state
    x_pos = self.robot_manager.entity.get_pos()[:, 0]
    s["max_x"] = torch.maximum(s["max_x"], x_pos)
    foot_forces = self.foot_contact_manager.contacts.norm(dim=-1)  # (N, 8)
    s["flight_steps"] += (foot_forces.max(dim=-1).values < 1.0).int()
    ep["Metrics / max_jump_distance"] = s["max_x"].mean().clone()
    ep["Metrics / mean_flight_steps"] = s["flight_steps"].float().mean().clone()

    done = terminated | truncated
    if done.any():
        s["max_x"][done] = 0.0
        s["flight_steps"][done] = 0

    return obs, reward, terminated, truncated, extras
```

### Rules
  - Write to `extras["episode"]` (or `env.extras["episode"]`) — never to bare `extras`
  - Values MUST be scalar tensors — call `.mean()` or `[0]` to get shape `()` before storing
  - Always call `.clone()` to avoid stale-reference bugs
  - The `"Metrics / "` prefix groups entries into a readable TensorBoard section

## Suggested metrics to log
  ep["Metrics / feet_airborne_frac"]   — fraction of 8 feet off ground per step (0–1)
  ep["Metrics / all_feet_off_frac"]    — fraction of envs in true full-airborne flight
  ep["Metrics / mean_forward_vel"]     — mean CoM x-velocity
  ep["Metrics / mean_upward_vel"]      — mean CoM z-velocity
  ep["Metrics / max_jump_distance"]    — max x-displacement reached this episode (per env, averaged)
  ep["Metrics / mean_flight_steps"]    — steps spent fully airborne this episode
  ep["Metrics / mean_launch_vx"]       — forward velocity at takeoff
  ep["Metrics / mean_launch_vz"]       — upward velocity at takeoff
  ep["Metrics / mean_landing_force"]   — peak non-foot body contact force at landing (N)
  ep["Metrics / clean_landing_rate"]   — fraction of landings with peak force < 15 N

## Code constraints
- Only import from `genesis_forge`, `spiderbot`, `torch`, and stdlib
- Maintain the `config()` and `build()` method signatures exactly
- You MAY add or extend `step()` in environment.py to log episodic metrics (see above)
- PRESERVE these managers exactly as-is (required by eval and training infrastructure):
    - `self.body_terrain_contact` — required by eval harness
    - `self.foot_contact_manager` with `track_air_time=True` — required by eval harness
    - `self.self_contact` — used for reward and monitoring
    - `self.actuator_manager`, `self.action_manager`, `self.robot_manager` — core infrastructure
- Do NOT add `self_contact` to TerminationManager
- Do NOT add `height_command_manager` — jumping env has none

## Response format
Return ONLY file content blocks in this exact format:

--- environment.py ---
<complete Python file content>
--- end ---

You may also include any of these blocks (only include files you actually want to change):
- `--- rewards.py ---` — custom reward functions
- `--- terminations.py ---` — custom termination functions
- `--- ppo.yaml ---` — hyperparameter changes

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


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|^```\n", re.MULTILINE)


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
        if name.lower() == "end":
            continue
        raw = parts[i + 1]
        # Strip a trailing `--- end ---` if present
        raw = _END_MARKER_RE.split(raw)[0]
        raw = raw.strip()
        # Strip markdown code fences the LLM often wraps around file content
        # e.g. ```python\n...\n```
        raw = _CODE_FENCE_RE.sub("", raw)
        raw = re.sub(r"\n```\s*$", "", raw).strip()
        blocks[name] = raw
    return blocks


# ---------------------------------------------------------------------------
# Cross-file import validation
# ---------------------------------------------------------------------------

_CUSTOM_MODULES = {
    "spiderbot.jumping.rewards": "rewards.py",
    "spiderbot.jumping.terminations": "terminations.py",
}


def _check_cross_file_imports(
    proposed: dict[str, str],
    current: dict[str, str],
) -> list[str]:
    """
    Verify that every name imported from spiderbot.jumping.rewards or
    spiderbot.jumping.terminations in environment.py actually exists in
    the corresponding file (using proposed content when available, otherwise
    falling back to the current on-disk content).
    """
    def effective(name: str) -> str:
        return proposed.get(name, current.get(name, ""))

    def top_level_names(src: str) -> set[str]:
        names: set[str] = set()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return names
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
        return names

    env_src = effective("environment.py")
    if not env_src:
        return []

    try:
        env_tree = ast.parse(env_src)
    except SyntaxError:
        return []  # syntax errors reported separately

    errors: list[str] = []
    for node in ast.walk(env_tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        # Support both absolute ("spiderbot.jumping.rewards") and relative (".rewards")
        if node.level > 0 and module in ("rewards", "terminations"):
            filename = module + ".py"
        elif module in _CUSTOM_MODULES:
            filename = _CUSTOM_MODULES[module]
        else:
            continue

        defined = top_level_names(effective(filename))
        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name not in defined:
                errors.append(
                    f"environment.py imports '{name}' from {filename}, "
                    f"but '{name}' is not defined there. "
                    f"Add the function to {filename} or remove the import."
                )
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_modifications(
    current_files: dict[str, str],
    run_history: list[dict],
    prompt_mode: Literal["explore", "tune"],
    model: str = "claude-sonnet-4-6",
    commentary: str = "",
) -> ProposalResult | None:
    """
    Call Claude to propose modifications to the jumping environment.

    Args:
        current_files: dict mapping filename → content for the files Claude may modify.
        run_history: Full run history list; only the last 5 entries are sent.
        prompt_mode: "explore" → qualitatively new reward structure;
                     "tune" → adjust weights only within the current structure.
        model: Anthropic model ID.
        commentary: Optional free-text observations from the operator (e.g. video notes,
                    chart observations) injected into the prompt before the file contents.

    Returns:
        ProposalResult with modified_files and reasoning, or None if all retries fail.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    user_message = _build_user_message(current_files, run_history, prompt_mode, commentary)
    client = anthropic.Anthropic()

    for attempt in range(2):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=32000,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                response = stream.get_final_message()
        except anthropic.APIError:
            raise

        # Warn if the response was cut off
        stop_reason = response.stop_reason
        if stop_reason == "max_tokens":
            _log.warning("LLM response was truncated (hit max_tokens) — increasing max_tokens may help")

        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block is None:
            _log.warning("LLM response contained no text block (attempt %d/2). Content types: %s",
                         attempt + 1, [b.type for b in response.content])
            continue
        text = text_block.text
        blocks = _parse_blocks(text)

        reasoning = blocks.pop("reasoning", "")
        if not reasoning:
            first_block_pos = text.find("---")
            if first_block_pos > 0:
                reasoning = text[:first_block_pos].strip()

        # A response with no file blocks is useless — treat as failure
        py_blocks = {k: v for k, v in blocks.items() if k.endswith(".py")}
        if not py_blocks:
            _log.warning(
                "LLM response contained no Python file blocks (attempt %d/2). "
                "stop_reason=%s. Response tail:\n%s",
                attempt + 1, stop_reason, text[-500:],
            )
            if attempt == 0:
                user_message = (
                    user_message
                    + "\n\nYour previous response contained no file blocks. "
                    "You MUST include at least one `--- environment.py ---` / `--- end ---` block "
                    "with the complete modified Python file."
                )
            continue

        # Validate Python syntax; accumulate errors for retry
        errors: list[str] = []
        for filename, content in py_blocks.items():
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{filename}: {e}")

        # Cross-file import check: names imported from rewards.py / terminations.py
        # in environment.py must actually be defined in those files.
        if not errors:
            errors = _check_cross_file_imports(blocks, current_files)

        if not errors:
            return ProposalResult(
                modified_files=blocks,
                reasoning=reasoning,
                mode=prompt_mode,
            )

        _log.warning("LLM response had errors (attempt %d/2): %s", attempt + 1, errors)
        if attempt == 0:
            error_text = "\n".join(errors)
            user_message = (
                user_message
                + f"\n\nYour previous response had errors. Fix them and try again:\n{error_text}"
            )

    return None


def _build_user_message(
    current_files: dict[str, str],
    run_history: list[dict],
    prompt_mode: Literal["explore", "tune"],
    commentary: str = "",
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

    if commentary.strip():
        lines.append(
            "\n## Operator observations\n"
            "The following notes were provided by the human operator based on reviewing eval videos "
            "and training charts. Treat these as high-priority constraints on your proposal:\n\n"
            + commentary.strip()
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
