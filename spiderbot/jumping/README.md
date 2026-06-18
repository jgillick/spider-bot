# Autonomous Jumping Agent

An AI agent that teaches the spider robot to jump by autonomously iterating on reward functions. It proposes reward modifications using Claude, runs short probe training sessions to screen candidates, promotes promising ones to full training runs, evaluates the resulting policy, and loops until the robot achieves a clean forward jump.

## How it works

```
LLM proposes reward changes
        ↓
Probe run (~600 iterations, 512 envs)
        ↓
Reward curve flat, diverging, or errored? → Move to failed/ → LLM tries again
        ↓
Probe looks promising?
        ↓
Full run (~1500 iterations, 2096 envs)
        ↓
Eval rollout → jump metrics collected
        ↓
Metrics meet success criteria? → Done!
        ↓
Not there yet → LLM tries again
```

Each iteration is a self-contained directory under `experiments/` containing all source files, logs, reasoning, and results. Scrapped experiments are moved to `failed/`. Nothing is ever deleted.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set your Anthropic API key

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com/).

The `.env` file is gitignored — never commit your key.

---

## Running the agent

```bash
python -m spiderbot.jumping.run_agent
```

The agent runs continuously until it finds a successful policy (or you stop it with `Ctrl+C`). On interrupt it saves `run_history.json` and exits cleanly — no files are modified or deleted.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--probe-iterations N` | 600 | Training iterations per probe run |
| `--full-iterations N` | 1500 | Training iterations for promoted full runs |
| `--probe-envs N` | 512 | Parallel environments during probes |
| `--full-envs N` | 2096 | Parallel environments for full runs |
| `--model MODEL` | `claude-sonnet-4-6` | Claude model to use for reward proposals |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--resume` | off | Resume from an existing `run_history.json` |

### Examples

Resume after an interruption:

```bash
python -m spiderbot.jumping.run_agent --resume
```

Use a more capable model:

```bash
python -m spiderbot.jumping.run_agent --model claude-opus-4-8
```

---

## Monitoring progress

### Live training output

Each probe and full run streams stdout to the terminal and also writes it to a log file inside the experiment directory:

```
experiments/iter0001_20260611_162020/
  0_probe_stdout.log
  1_full_stdout.log
```

### TensorBoard

```bash
tensorboard --logdir spiderbot/jumping/experiments
```

### Run history

After each iteration, `run_history.json` is updated:

```bash
cat spiderbot/jumping/run_history.json | python -m json.tool | less
```

Each entry contains:
- `iteration` — iteration number
- `iter_dir` — experiment directory name (e.g. `iter0001_20260611_162020`)
- `run_type` — `probe` or `full`
- `stop_reason` — `completed`, `flatline`, `divergence`, or `error`
- `is_promising` — whether the run met the jump quality threshold
- `metrics` — jump metrics from the eval harness (full runs only)
- `reasoning_summary` — first 500 chars of the LLM's reasoning

### Experiment directories

Every iteration creates a directory under `experiments/`:

```
experiments/
  base/                         ← immutable template (never modified by the agent)
    environment.py
    rewards.py
    terminations.py
    ppo.yaml
    train.py
    eval.py

  iter0001_20260611_162020/     ← one directory per iteration
    environment.py              ← LLM-proposed reward code
    rewards.py
    terminations.py
    ppo.yaml
    train.py
    eval.py
    reasoning.md                ← LLM's explanation of what it changed and why
    evaluation.md               ← probe/full/eval summary written after the run
    metrics.json                ← raw numbers (probe reward curve + eval metrics)
    logs/
      0_probe/                  ← probe training logs + checkpoints
      1_full/                   ← full training logs + checkpoints (if promoted)

failed/
  iter0002_20260611_163113/     ← scrapped runs moved here (probe failed or no checkpoint)
```

Experiments that complete a full eval (promising or not) stay in `experiments/`. Experiments whose probe run failed or produced no checkpoint are moved to `failed/`.

---

## Re-running eval on a specific iteration

If an eval step failed or you want to re-evaluate after changing thresholds:

```bash
python -m spiderbot.jumping.run_eval iter0001_20260611_162020
python -m spiderbot.jumping.run_eval iter0001_20260611_162020 --device cpu --no-video
```

This updates `evaluation.md` in the experiment directory.

---

## Success criteria

The agent declares success when a policy achieves all three (thresholds in `config.py`):

1. All 8 feet simultaneously off the ground for at least 1 step
2. Positive forward (x-axis) displacement during the jump
3. Peak contact force on non-foot body links at landing below **50 N**

---

## Stopping and restarting

**Clean stop:** Press `Ctrl+C`. The agent saves `run_history.json` and exits. No files are modified or deleted.

**Resume:** Pass `--resume` to pick up from the last completed iteration. The LLM receives the full run history as context.

**Fresh start:** Delete (or rename) `run_history.json` and omit `--resume`.

---

## Configuration

Edit `config.py` to change thresholds and training schedule defaults without touching any other file:

```python
# Force threshold for a clean landing (N) — 50 N ≈ 11 lbs
SUCCESS_FORCE_THRESHOLD_N = 50.0

# Minimum CoM height gain to confirm feet left the ground
SUCCESS_HEIGHT_THRESHOLD_M = 0.02

# Whether to carry a run forward as the LLM's next base
PROMISING_FORCE_N = 50.0   # can be set looser than SUCCESS_FORCE_THRESHOLD_N

# Training schedule
PROBE_ITERATIONS = 600
FULL_ITERATIONS  = 1500
PROBE_NUM_ENVS   = 512
FULL_NUM_ENVS    = 2096
```

All CLI flags override the defaults in `config.py` for a single run.

---

## Editing the LLM prompt

The system prompt sent to Claude on every iteration lives in `prompt.md`. Edit it directly — no Python changes needed. Two values are interpolated at load time from `config.py`:

- `${SUCCESS_FORCE_THRESHOLD_N}` — landing force threshold shown to the LLM
- `${SUCCESS_HEIGHT_THRESHOLD_M}` — minimum height threshold shown to the LLM

---

## File overview

| File | Purpose |
|------|---------|
| `run_agent.py` | CLI entry point |
| `agent.py` | Main orchestration loop |
| `llm_engine.py` | Claude API integration, reward proposal and validation |
| `runner.py` | Launches training subprocesses, monitors reward curves |
| `snapshot.py` | `is_promising()` helper used by agent and run_eval |
| `config.py` | All tunable thresholds and training schedule defaults |
| `prompt.md` | System prompt sent to Claude — edit to change LLM behaviour |
| `run_eval.py` | CLI to re-run eval on a specific experiment iteration |
| `run_history.json` | Runtime artifact — full iteration log |
| `experiments/` | Runtime artifact — per-iteration experiment directories |
| `experiments/base/` | Immutable template copied at the start of each iteration |
| `failed/` | Runtime artifact — scrapped experiment directories |
