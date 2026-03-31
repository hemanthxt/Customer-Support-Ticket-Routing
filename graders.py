"""
graders.py
Rule-based, deterministic graders for each task.
Every grader returns a float in [0.0, 1.0].

Scoring breakdown (same for all tasks):
  category correct   → +0.30
  priority correct   → +0.20
  team     correct   → +0.20
  reply quality      → +0.20  (keyword hit-rate)
  final handling     → +0.10  (escalation / resolve correctness)
"""

from __future__ import annotations
from typing import Any


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _reply_score(reply: str, keywords: list[str]) -> float:
    """Return fraction of expected keywords found in reply (case-insensitive)."""
    if not reply:
        return 0.0
    reply_lower = reply.lower()
    hits = sum(1 for kw in keywords if kw in reply_lower)
    # Give full marks when ≥50 % keywords are present
    fraction = hits / len(keywords)
    return min(fraction * 2, 1.0)          # scale: 50 % hits → 1.0


def _bool_score(agent_val: bool, expected_val: bool) -> float:
    return 1.0 if agent_val == expected_val else 0.0


# ──────────────────────────────────────────────────────────────
# Generic grader  (works for all 3 tasks)
# ──────────────────────────────────────────────────────────────

def grade(state: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """
    Score an episode given final ticket state and the task's expected dict.

    Returns a breakdown dict with individual components and a 'total' key.
    """

    scores: dict[str, float] = {}

    # 1. Category  (30 %)
    agent_cat    = state.get("category")
    expected_cat = expected["category"]
    if hasattr(expected_cat, "value"):
        expected_cat = expected_cat.value
    scores["category"] = 0.30 if str(agent_cat) == str(expected_cat) else 0.0

    # 2. Priority  (20 %)
    agent_pri    = state.get("priority")
    expected_pri = expected["priority"]
    if hasattr(expected_pri, "value"):
        expected_pri = expected_pri.value
    scores["priority"] = 0.20 if str(agent_pri) == str(expected_pri) else 0.0

    # 3. Team  (20 %)
    agent_team    = state.get("assigned_team")
    expected_team = expected["team"]
    if hasattr(expected_team, "value"):
        expected_team = expected_team.value
    scores["team"] = 0.20 if str(agent_team) == str(expected_team) else 0.0

    # 4. Reply quality  (20 %)
    reply    = state.get("reply_draft", "")
    keywords = expected.get("reply_keywords", [])
    scores["reply"] = round(0.20 * _reply_score(reply, keywords), 4)

    # 5. Final handling  (10 %)
    agent_escalated = state.get("escalated", False)
    agent_resolved  = state.get("status") == "resolved"

    if expected.get("should_escalate"):
        scores["handling"] = round(0.10 * _bool_score(agent_escalated, True), 4)
    else:
        scores["handling"] = round(0.10 * _bool_score(agent_resolved, expected.get("should_resolve", True)), 4)

    # Total
    scores["total"] = round(sum(scores.values()), 4)

    return scores


# ──────────────────────────────────────────────────────────────
# Task-specific wrappers (for explicit clarity in reports)
# ──────────────────────────────────────────────────────────────

def grade_task_01(state: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """Grade the Easy billing task."""
    return grade(state, expected)


def grade_task_02(state: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """Grade the Medium account-access task."""
    return grade(state, expected)


def grade_task_03(state: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """Grade the Hard multi-issue task."""
    return grade(state, expected)


GRADER_MAP = {
    "TASK_01_EASY":   grade_task_01,
    "TASK_02_MEDIUM": grade_task_02,
    "TASK_03_HARD":   grade_task_03,
}


def get_grader(task_id: str):
    if task_id not in GRADER_MAP:
        raise ValueError(f"No grader for task_id: {task_id!r}")
    return GRADER_MAP[task_id]
