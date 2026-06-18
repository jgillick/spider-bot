"""
LLM reasoning engine for the autonomous jumping agent.
Calls Claude to propose modifications to the jumping environment files,
validates the generated Python, and returns structured results.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Literal

import anthropic

from .config import SUCCESS_FORCE_THRESHOLD_N, SUCCESS_HEIGHT_THRESHOLD_M

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ProposalResult:
    modified_files: dict[str, str]
    reasoning: str
    mode: str


# ---------------------------------------------------------------------------
# System prompt — loaded from prompt.md and interpolated with config values
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_prompt() -> str:
    with open(os.path.join(_THIS_DIR, "prompt.md")) as f:
        text = f.read()
    text = text.replace("${SUCCESS_FORCE_THRESHOLD_N}", str(SUCCESS_FORCE_THRESHOLD_N))
    text = text.replace("${SUCCESS_HEIGHT_THRESHOLD_M}", str(SUCCESS_HEIGHT_THRESHOLD_M))
    return text

_SYSTEM_PROMPT = _load_prompt()


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
