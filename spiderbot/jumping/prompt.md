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
- CoM height must exceed resting height by at least ${SUCCESS_HEIGHT_THRESHOLD_M} m (minimum jump confirmation — feet must leave the ground; this is not a height goal)
- Peak contact force on non-feet body links must stay below ${SUCCESS_FORCE_THRESHOLD_N} N at landing
- **No early terminations** — the 250-step eval rollout must complete without any termination condition firing (body slam, bad orientation, etc.). A policy that jumps but crashes on landing fails this criterion.

## Available managers in self (ONLY use what is listed here)
These are the attributes you can reference in params:
  - self.actuator_manager   — ActuatorManager (joint torques, positions, velocities)
  - self.action_manager     — PositionActionManager (raw actions, DOF positions/velocities)
  - self.robot_manager      — EntityManager (CoM position, linear/angular velocity)
  - self.foot_contact_manager — ContactManager, track_air_time=True (foot-to-terrain contacts)
  - self.self_contact        — ContactManager (robot-to-robot self contacts)
  - self.body_terrain_contact — ContactManager (non-foot body-to-terrain contacts; DO NOT REMOVE)

## Reward function reference (genesis_forge.mdp.rewards)
Use ONLY these exact names with ONLY the listed params — any unknown keyword argument crashes at runtime.
For behaviors not expressible here, write a custom function in `rewards.py` (see below).

Each entry shows the exact `"params"` dict to use in the reward config.

**Alive / termination**
  - `rewards.is_alive` — 1.0 while the robot is alive
    params: {}
  - `rewards.terminated` — 1.0 on the termination step
    params: {}

**Joint effort and smoothness**
  - `rewards.dof_torque_l2` — L2 penalty on joint torques
    params: {"actuator_manager": self.actuator_manager}
  - `rewards.dof_velocity_l2` — L2 penalty on joint velocities
    params: {"action_manager": self.action_manager}
  - `rewards.dof_similar_to_default` — penalty for deviation from the default joint pose
    params: {"actuator_manager": self.actuator_manager}   ← or action_manager, not both
  - `rewards.action_rate_l2` — penalty on action change between steps
    params: {}
  - `rewards.action_acceleration_l2` — penalty on second derivative of actions (oscillation/jitter)
    params: {}   ← MdpFnClass: reference the class directly as `"fn": rewards.action_acceleration_l2`

**Body motion**
  - `rewards.lin_vel_z_l2` — L2 penalty on vertical CoM velocity
    params: {}
  - `rewards.ang_vel_xy_l2` — L2 penalty on body roll/pitch rate
    params: {}
  - `rewards.lin_vel_xy_l2` — L2 penalty on horizontal CoM velocity
    params: {"entity_manager": self.robot_manager}   ← entity_manager is REQUIRED; no default
  - `rewards.body_acceleration_exp` — exponential penalty on body acceleration; rewards smooth motion
    params: {}   ← MdpFnClass; optional override: {"sensitivity": 0.05} — lower = harsher penalty

**Contact**
  - `rewards.contact_force` — penalty proportional to contact force above threshold
    params: {"contact_manager": self.self_contact}   ← or self.body_terrain_contact
    optional: add "threshold": 5.0 to change the floor (default 1.0 N)
  - `rewards.feet_air_time` — reward for feet spending continuous time airborne
    params: {"contact_manager": self.foot_contact_manager, "time_threshold": 0.2}
    ⚠️ contact_manager MUST have track_air_time=True — only self.foot_contact_manager qualifies
    optional: add "time_threshold_max": 1.0 to cap the per-foot reward
    ⚠️ This reward is per-foot and independent — a robot standing tall on 3 legs with 5 feet raised
    will score well without ever achieving a true jump. If used, pair with a custom all-feet-simultaneous
    liftoff check (see custom rewards below) to prevent this exploit.

## INTERNAL UTILITIES — NOT reward functions (DO NOT USE in reward config)
entity_lin_vel, entity_ang_vel, entity_projected_gravity are internal helpers.
They take a RigidEntity, not an env. Do NOT put them in the RewardManager config.

## Excluded reward terms (MUST NOT use — they actively oppose jumping)
  - flat_orientation_l2   — penalises body tilt needed for jump takeoff
  - base_height           — targeting resting height prevents lifting off
  - has_contact           — rewards feet staying planted on the ground
  - command_tracking_lin_vel / command_tracking_ang_vel — locomotion gait terms
  - feet_ground_time      — penalises feet leaving the ground
  - CoM z-position (height) as a positive reward — causes the robot to stand as tall as possible
    on a subset of legs rather than jump. The eval requires ALL 8 feet simultaneously off the
    ground; a policy that lifts 5 feet while 3 remain planted will always fail eval.

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

Known failure mode — "jump-then-crash": The robot achieves liftoff (all feet off ground) but triggers
an early termination on landing (body slam, bad orientation, etc.), resetting the episode before any
forward distance is recorded. Seen in multiple runs: `airborne_steps > 0` but `early_terminations > 0`
and `forward_distance_m ≈ 0`. Fix: relax or remove aggressive termination conditions, or add a reward
for soft landing so the policy learns to absorb the impact. The `contact_force_with_grace_period`
termination is preferred over `contact_force` because it gives the robot a brief window to settle.

Known failure mode — "tall standing": Rewarding CoM height or using `feet_air_time` without an
all-feet-simultaneous check causes the robot to stand as tall as possible on a subset of legs.
Observed in multiple runs: robot stood high on 3 legs with 5 feet raised — satisfying height and
partial air-time rewards without ever leaving the ground with all 8 feet. The eval requires ALL 8
feet simultaneously off the ground; any reward that can be maximised by lifting only some feet will
produce this behaviour.

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

### history_len — temporal context

Pass `history_len=N` to `ObservationManager` to stack the last N observations into each policy input.
This gives the policy temporal context — useful for detecting liftoff, tracking trajectory, and timing
the landing crouch. Comes at the cost of larger input size (N × obs_dim).

```python
ObservationManager(
    self,
    history_len=3,   # stack last 3 steps; try 2–5 for jumping
    cfg={...},
)
```

With history, the policy can infer per-step changes (approximate velocity/acceleration) without needing
explicit velocity observations. Omit or set to 1 to disable (default behaviour).

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

### Privileged critic observations (asymmetric actor-critic)

The critic only runs during training — it can observe privileged ground-truth state that the deployed
policy cannot. Better critic inputs → better advantage estimates → faster learning. This is standard
practice for contact-rich robot tasks.

To add privileged observations, create a **second** `ObservationManager` with `name="critic"` — it
must be a separate instance, NOT a nested group inside the existing cfg dict:

**Step 1** — add a second `ObservationManager` in `config()`:
```python
# Existing actor observations — keep as-is (name defaults to "policy")
ObservationManager(
    self,
    cfg={
        "imu_lin_acc_ang_vel": {"fn": self.imu_observation},
        # ... rest of existing observations unchanged ...
    },
)

# Second manager — critic-only privileged observations
ObservationManager(
    self,
    name="critic",   # ← this is what links it to ppo.yaml
    cfg={
        "true_foot_forces": {
            "fn": lambda env: env.foot_contact_manager.contacts.norm(dim=-1),
        },
        "airborne": {
            "fn": lambda env: (
                env.foot_contact_manager.contacts.norm(dim=-1).max(dim=-1).values < 1.0
            ).float().unsqueeze(-1),
        },
        "body_contact_forces": {
            "fn": lambda env: env.body_terrain_contact.contacts.norm(dim=-1),
        },
        "com_height": {
            "fn": lambda env: env.robot_manager.entity.get_pos()[:, 2:3],
        },
    },
)
```

**Step 2** — update `ppo.yaml` so the critic sees both groups:
```yaml
obs_groups:
  actor:
    - policy
  critic:
    - policy
    - critic    # ← add this line
```

Do NOT put the critic observations inside the `policy` manager's `cfg` — the deployed policy cannot
access ground-truth physics state. Each `ObservationManager` instance is one flat group.

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

## Actuator and action configuration — you MAY modify

The `ActuatorManager` and `PositionActionManager` in `config()` control joint stiffness, torque limits, and how policy actions map to joint targets. These are worth tuning when the robot lacks the power or speed to jump, or when leg motion is too oscillatory/sluggish.

### ActuatorManager parameters

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=".*",
    default_pos={...},  # default standing pose — see below
    kp=50.0,            # position gain (stiffness)
    kv=0.5,             # velocity gain (damping)
    max_force=18.0,    # peak torque per joint (N·m)
    frictionloss=...,   # DO NOT CHANGE
    damping=...,        # DO NOT CHANGE
    armature=...,       # DO NOT CHANGE
)
```

| Parameter | Current | Suggested range | Effect on jumping |
|-----------|---------|-----------------|-------------------|
| `kp` | 50.0 | 20–150 | Higher → stiffer legs, more explosive extension, but not as soft for landing |
| `kv` | 0.5 | 0.1–2.0 | Higher → more damped, less oscillation at cost of speed |
| `max_force` | 18.0 | 5.0 - 25.0 | Higher → more torque available for the jump |

`NoisyValue(mean, std)` adds per-env Gaussian noise at reset — keep the wrapper when changing `max_force`.

**Do NOT change** `frictionloss`, `damping`, or `armature` — these are physics-calibrated sim2real parameters.

### default_pos — default standing pose

`default_pos` sets the joint angles the robot resets to, and the zero-point that policy actions are offsets from. The current values produce a stable standing pose. Modifying them to a pre-coiled/crouched posture (e.g. more femur flexion, deeper tibia angle) can give the legs more range to extend explosively at takeoff. Keep joint names consistent with the existing pattern (`R1_Hip`, `[RL][1-4]_Femur`, etc.).

### PositionActionManager parameters

```python
self.action_manager = PositionActionManager(
    self,
    scale=0.25,                   # max joint position delta per action step (rad)
    soft_limit_scale_factor=0.95, # fraction of hard joint limits used as soft limits
    ...
)
```

| Parameter | Current | Suggested range | Effect on jumping |
|-----------|---------|-----------------|-------------------|
| `scale` | 0.25 | 0.15–0.5 | Higher → larger joint displacement per step, faster kick |
| `soft_limit_scale_factor` | 0.95 | 0.85–1.0 | Higher → more usable joint range (1.0 = full hard-limit range) |

---

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
  ep["Metrics / clean_landing_rate"]   — fraction of landings with peak force < ${SUCCESS_FORCE_THRESHOLD_N} N

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

## PPO configuration — you MAY modify

The `ppo.yaml` file controls the training algorithm. Return a `--- ppo.yaml ---` block to change it.
Include the full file — RSL-RL reads it wholesale.

Key parameters for jumping:

| Parameter | Current | Effect |
|-----------|---------|--------|
| `entropy_coef` | 0.005 | Higher (0.01–0.05) → more exploration; useful when the robot is stuck on the ground |
| `num_steps_per_env` | 24 | Steps collected per env before each update; higher (48–96) gives more signal for delayed rewards like jump landings |
| `gamma` | 0.995 | Discount factor — already tuned for jumping's delayed payout; do not lower |
| `learning_rate` | 0.001 | Adaptive schedule (`schedule: adaptive`) adjusts this automatically via `desired_kl` |
| `hidden_dims` (actor/critic) | [1024, 512, 256] | Network capacity; increase if the task is too complex for current architecture |
| `init_std` | 1.0 | Initial policy action std; higher = more random early exploration |

### RND — intrinsic exploration bonus

The config includes a `rnd_cfg` block (Random Network Distillation). When enabled, it adds a bonus
for visiting novel states — useful when the robot is stuck and rarely discovers the airborne state.

To enable:
```yaml
rnd_cfg:
  weight: 0.01     # intrinsic reward weight; try 0.005–0.05
  weight_schedule: null
  reward_normalization: true
```

Start small (`weight: 0.01`) and watch TensorBoard — too high and the robot chases novelty instead
of learning to jump. Disable once the robot is reliably leaving the ground.

---

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
