# Jumping agent configuration — edit these values to tune behaviour.

# ---------------------------------------------------------------------------
# Eval thresholds
# ---------------------------------------------------------------------------

# Peak contact force (N) on non-feet links above which a landing counts as
# a hard body impact (failure).  ~50 N ≈ 11 lbs of force.
SUCCESS_FORCE_THRESHOLD_N: float = 75.0

# Contact force (N) per foot below which a foot is considered airborne.
AIRBORNE_FORCE_THRESHOLD_N: float = 1.0

# Minimum CoM height gain (m) required to confirm feet left the ground.
# This is a floor check, not a performance goal.
SUCCESS_HEIGHT_THRESHOLD_M: float = 0.02

# Rollout length for evaluation (steps at 50 Hz → 5 s of robot time).
EVAL_NUM_STEPS: int = 250

# ---------------------------------------------------------------------------
# Promising-run thresholds
# Used to decide whether to carry an iteration forward as the next LLM base.
# Can be set looser than the success thresholds to preserve experiments with
# good jump mechanics even if the landing isn't perfect.
# ---------------------------------------------------------------------------

PROMISING_HEIGHT_M: float = SUCCESS_HEIGHT_THRESHOLD_M
PROMISING_FORCE_N: float = SUCCESS_FORCE_THRESHOLD_N

# ---------------------------------------------------------------------------
# Training schedule defaults
# All of these can be overridden via CLI flags on run_agent.py.
# ---------------------------------------------------------------------------

PROBE_ITERATIONS: int = 600
PROBE_NUM_ENVS: int = 512
FULL_ITERATIONS: int = 1500
FULL_NUM_ENVS: int = 3072

# Maximum consecutive LLM proposal failures before the agent aborts.
MAX_LLM_FAILURES: int = 3
