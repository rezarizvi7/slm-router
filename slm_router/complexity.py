"""Cheap, deterministic prompt-complexity scoring.

This runs on every request, so it has to cost microseconds, not milliseconds. It
produces a score in [0, 1] plus human-readable reasons so routing decisions are
debuggable (they are also returned in the X-Router-Reasons response header).

The weights are hand-tuned starting points, not learned. Use scripts/bench.py on a
labelled sample of your own traffic to tune them and the threshold.
"""

import math
import re
from dataclasses import dataclass, field

# Signals that usually mean "multi-step reasoning or generation the SLM will fumble".
HARD_PATTERNS: dict[str, tuple[re.Pattern, float]] = {
    "multi_step": (re.compile(r"\b(step[- ]by[- ]step|first,? .{0,80}\bthen\b|and then|after that|finally,)", re.I), 0.9),
    "design": (re.compile(r"\b(architect\w*|system design|design (a|an|the)|trade-?offs?|scalab\w+)\b", re.I), 1.1),
    "reasoning": (re.compile(r"\b(prove|derive|proof|why does|explain why|reason about|root cause|counterexample)\b", re.I), 1.0),
    "code_heavy": (re.compile(r"\b(refactor|implement|debug|optimi[sz]e|race condition|deadlock|concurren\w+|migrat\w+)\b", re.I), 0.8),
    "analysis": (re.compile(r"\b(analy[sz]e|compare|evaluate|critique|review|assess)\b", re.I), 0.6),
    "math": (re.compile(r"(\\frac|\\sum|\\int|∑|∫|\b(integral|eigen\w*|theorem|lemma|complexity of)\b)", re.I), 0.9),
}

# Signals that usually mean "short, well-scoped task a 1-3B model handles fine".
EASY_PATTERNS: dict[str, tuple[re.Pattern, float]] = {
    "lookup": (re.compile(r"^\s*(what is|what's|who is|define|when (was|did)|how do i say)\b", re.I), -0.9),
    "transform": (re.compile(r"\b(translate|rephrase|reword|fix (the )?(typo|grammar|spelling)|format|convert|rename|capitali[sz]e)\b", re.I), -0.8),
    "short_gen": (re.compile(r"\b(one[- ]liner|commit message|regex|title for|tl;?dr|summari[sz]e (this|in one))\b", re.I), -0.7),
    "classify": (re.compile(r"\b(classify|categori[sz]e|sentiment|yes or no|true or false|extract the)\b", re.I), -0.7),
}

NUMBERED_STEP = re.compile(r"^\s*(\d+[.)]|[-*])\s+\S", re.M)
CODE_FENCE = re.compile(r"```")


@dataclass
class ComplexityResult:
    score: float
    reasons: list[str] = field(default_factory=list)
    # Hard overrides bypass the score entirely (e.g. tool calling, images).
    force_frontier: bool = False


def message_text(content) -> str:
    """Flatten OpenAI-style content (str or list of parts) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(parts)


def _has_image(messages: list[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "image") for p in c):
            return True
    return False


def score(messages: list[dict], tools: list | None = None) -> ComplexityResult:
    reasons: list[str] = []

    if tools:
        return ComplexityResult(1.0, ["tools_present"], force_frontier=True)
    if _has_image(messages):
        return ComplexityResult(1.0, ["image_input"], force_frontier=True)

    user_turns = [m for m in messages if m.get("role") == "user"]
    last_user = message_text(user_turns[-1]["content"]) if user_turns else ""
    all_text = "\n".join(message_text(m.get("content")) for m in messages)

    # Logit starts slightly negative: default to local unless there's evidence.
    logit = -0.6

    # Length of the latest ask (≈4 chars/token). log-scaled so a 4k-token paste
    # doesn't saturate on its own: ~+0 at 50 tok, ~+1.2 at 800 tok, ~+1.8 at 3k tok.
    approx_tokens = max(1, len(last_user) // 4)
    length_term = 0.45 * math.log2(approx_tokens / 50) if approx_tokens > 50 else 0.0
    if length_term:
        logit += length_term
        reasons.append(f"length~{approx_tokens}tok")

    # Total context the model has to hold (long multi-turn chats get harder).
    ctx_tokens = len(all_text) // 4
    if ctx_tokens > 6000:
        logit += 1.0
        reasons.append(f"long_context~{ctx_tokens}tok")

    if len(user_turns) > 6:
        logit += 0.4
        reasons.append(f"turns={len(user_turns)}")

    for name, (pat, w) in HARD_PATTERNS.items():
        if pat.search(last_user):
            logit += w
            reasons.append(name)
    for name, (pat, w) in EASY_PATTERNS.items():
        if pat.search(last_user):
            logit += w
            reasons.append(name)

    steps = len(NUMBERED_STEP.findall(last_user))
    if steps >= 3:
        logit += 0.25 * min(steps, 8)
        reasons.append(f"enumerated_items={steps}")

    questions = last_user.count("?")
    if questions >= 3:
        logit += 0.5
        reasons.append(f"questions={questions}")

    fences = len(CODE_FENCE.findall(last_user)) // 2
    if fences:
        logit += 0.4 * min(fences, 3)
        reasons.append(f"code_blocks={fences}")

    return ComplexityResult(score=1 / (1 + math.exp(-logit)), reasons=reasons)
