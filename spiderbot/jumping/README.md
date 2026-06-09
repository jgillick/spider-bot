# Autonomous Jumping Agent

An AI agent that teaches the spider robot to jump by autonomously iterating on reward functions. It proposes reward modifications using Claude, runs short probe training sessions to screen candidates, promotes promising ones to full training runs, evaluates the resulting policy, and loops until the robot achieves a clean forward jump.

## How it works

```
LLM proposes reward changes
        ↓
Probe run (500–800 iterations)
        ↓
Reward curve flat or diverging? → Scrap → LLM tries again
        ↓
Probe looks promising?
        ↓
Full run (~1500 iterations)
        ↓
Eval rollout → jump metrics collected
        ↓
Metrics meet success criteria? → Done!
        ↓
Not there yet → LLM tries again
```

All reward function code, training configs, and evaluation results are saved as snapshots under `experiments/` so you can inspect and replay any iteration.

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

The agent runs continuously until it finds a successful policy (or you stop it with `Ctrl+C`). On interrupt, it saves its progress and restores the original training files.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--probe-iterations N` | 600 | Training iterations per probe run |
| `--full-iterations N` | 1500 | Training iterations for promoted full runs |
| `--probe-envs N` | 512 | Parallel environments during probes (lower = faster per iter) |
| `--full-envs N` | 2096 | Parallel environments for full runs |
| `--model MODEL` | `claude-sonnet-4-6` | Claude model to use for reward proposals |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--resume` | off | Resume from an existing `run_history.json` |

### Examples

Run with faster probes for initial exploration:

```bash
python -m spiderbot.jumping.run_agent --probe-iterations 400 --probe-envs 256
```

Resume after an interruption:

```bash
python -m spiderbot.jumping.run_agent --resume
```

Use a more capable model if reward structure quality is insufficient:

```bash
python -m spiderbot.jumping.run_agent --model claude-opus-4-8
```

---

## Monitoring progress

### Live training output

Each probe and full run streams stdout to the terminal and also writes it to:

```
spiderbot/jumping/logs/<experiment_name>_stdout.log
```

### Run history

After each iteration, `run_history.json` is updated with a summary:

```bash
cat spiderbot/jumping/run_history.json | python -m json.tool | less
```

Each entry contains:
- `iteration` — iteration number
- `run_type` — `probe` or `full`
- `stop_reason` — `completed`, `flatline`, `divergence`, or `error`
- `is_promising` — whether the run met the jump quality threshold
- `metrics` — jump metrics from the eval harness (full runs only)
- `reasoning_summary` — first 500 chars of the LLM's reasoning

### Snapshots

Every run creates a snapshot under `experiments/`:

```
experiments/
  iter0001_probe_20260608_143022/
    environment.py   ← exact reward code that was trained
    ppo.yaml
    train.py
    metrics.json     ← reward curve summary + eval metrics
    reasoning.md     ← LLM's explanation of what it changed and why
  iter0002_full_20260608_151045/
    environment.py
    model_1500.pt    ← checkpoint (full runs only)
    metrics.json
    reasoning.md
  pruned/
    iter0001_probe_.../   ← scrapped runs: only metrics.json + reasoning.md kept
```

Snapshots for promising full runs are **never deleted** by the agent. Non-promising snapshots are pruned to save disk — their `metrics.json` and `reasoning.md` move to `experiments/pruned/`.

---

## Success criteria

The agent declares success when a policy achieves all three:

1. All 8 feet simultaneously off the ground for at least 1 step
2. Positive forward (x-axis) displacement during the jump
3. Peak contact force on non-foot body links at landing below 15 N

---

## Stopping and restarting

**Clean stop:** Press `Ctrl+C`. The agent will:
- Send SIGTERM to the current training subprocess
- Save `run_history.json`
- Restore `environment.py`, `ppo.yaml`, and `train.py` from `.bak` backups

**Resume:** Pass `--resume` to pick up from the last completed iteration. The LLM will receive the full run history as context.

**Fresh start:** Delete (or rename) `run_history.json` and omit `--resume`.

---

## File overview

| File | Purpose |
|------|---------|
| `run_agent.py` | CLI entry point — start here |
| `agent.py` | Main orchestration loop |
| `llm_engine.py` | Claude API integration, reward proposal and validation |
| `runner.py` | Launches training subprocesses, monitors reward curves |
| `eval.py` | Loads a checkpoint, runs a rollout, returns jump metrics |
| `snapshot.py` | Creates and prunes experiment snapshots |
| `environment.py` | The jumping RL environment (modified by the agent) |
| `train.py` | Training script (also importable as `train_main()`) |
| `ppo.yaml` | PPO hyperparameter config |
| `run_history.json` | Runtime artifact — full iteration log |
| `experiments/` | Runtime artifact — per-run snapshots |
