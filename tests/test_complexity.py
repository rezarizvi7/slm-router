from slm_router.complexity import score

EASY = [
    "What is a mutex?",
    "Translate 'good morning' to Spanish.",
    "Fix the typo: 'teh quick brown fox'",
    "Write a commit message for: bump lodash to 4.17.21",
    "Classify the sentiment of: 'I love this library'",
    "Convert 10 miles to km",
]

HARD = [
    "Design a multi-region rate limiter for 1M RPS. Discuss trade-offs between token bucket and sliding "
    "window, consistency across regions, and failure modes. Then propose a rollout plan step by step.",
    "Here is my code:\n```python\nimport threading\nlock = threading.Lock()\n```\nand\n```python\n"
    "def worker(): ...\n```\nDebug the deadlock and explain why it happens, then refactor it.",
    "Prove that the halting problem is undecidable and derive the consequence for Rice's theorem.",
    "Analyze these requirements and implement a plan:\n1. auth\n2. billing\n3. audit logs\n4. SSO\n5. RBAC",
]


def test_easy_prompts_score_low():
    for p in EASY:
        r = score([{"role": "user", "content": p}])
        assert r.score < 0.5, (p, r)


def test_hard_prompts_score_high():
    for p in HARD:
        r = score([{"role": "user", "content": p}])
        assert r.score >= 0.5, (p, r)


def test_tools_force_frontier():
    r = score([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "x"}}])
    assert r.force_frontier and "tools_present" in r.reasons


def test_images_force_frontier():
    msg = {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                       {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}
    assert score([msg]).force_frontier


def test_length_raises_score_monotonically():
    short = score([{"role": "user", "content": "summarize the doc " + "word " * 20}]).score
    long = score([{"role": "user", "content": "summarize the doc " + "word " * 2000}]).score
    assert long > short
