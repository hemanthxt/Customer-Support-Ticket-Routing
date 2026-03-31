"""
environment.py
Core TicketRoutingEnv — implements reset(), step(), state().

Reward signal breakdown per step:
  classify_ticket  correct  → +0.30  |  wrong → -0.10
  set_priority     correct  → +0.20  |  wrong → -0.05
  assign_team      correct  → +0.20  |  wrong → -0.10
  draft_reply      useful   → +0.20  |  empty → 0.00
  escalate         correct  → +0.10  |  unnecessary → -0.05
  resolve_ticket   correct  → +0.10  |  too early → -0.10
  repeated action           → -0.05  (same action_type called twice)
"""

from __future__ import annotations
from typing import Optional

from models import (
    Action, ActionType, Observation,
    StepResult, TicketState, TicketStatus,
)
from tasks import get_task, TASKS
from graders import get_grader


MAX_STEPS = 10          # episode ends after this many steps even if not resolved


class TicketRoutingEnv:
    """
    OpenEnv-compatible environment for customer-support ticket routing.

    Usage:
        env = TicketRoutingEnv()
        obs = env.reset("TASK_01_EASY")
        result = env.step(Action(action_type="classify_ticket", value="billing"))
        ...
    """

    def __init__(self) -> None:
        self._task: Optional[dict]        = None
        self._state: Optional[TicketState] = None

    # ─────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────

    def reset(self, task_id: str = "TASK_01_EASY") -> Observation:
        """Load a task and return the initial observation."""
        self._task = get_task(task_id)
        ticket     = self._task["ticket"]

        self._state = TicketState(
            ticket_id        = ticket["ticket_id"],
            customer_message = ticket["customer_message"],
            customer_tier    = ticket["customer_tier"],
            previous_tickets = ticket["previous_tickets"],
            status           = TicketStatus.open,
        )
        return self._build_observation()

    # ─────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Apply one action, return (observation, reward, done, info).
        action can be an Action instance or a plain dict.
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Call reset() before step().")

        # Accept dict as well as Action
        if isinstance(action, dict):
            action = Action(**action)

        s        = self._state
        expected = self._task["expected"]
        reward   = 0.0
        done     = False
        info: dict = {"action": action.action_type, "value": action.value}

        # ─── Repeated-action penalty ───
        if action.action_type in s.actions_taken:
            reward -= 0.05
            info["penalty"] = "repeated_action"
        else:
            s.actions_taken.append(action.action_type)

        # ─── Dispatch ───
        if action.action_type == ActionType.classify_ticket:
            reward += self._handle_classify(action, expected)

        elif action.action_type == ActionType.set_priority:
            reward += self._handle_priority(action, expected)

        elif action.action_type == ActionType.assign_team:
            reward += self._handle_assign(action, expected)

        elif action.action_type == ActionType.draft_reply:
            reward += self._handle_reply(action, expected)

        elif action.action_type == ActionType.escalate:
            reward += self._handle_escalate(action, expected)

        elif action.action_type == ActionType.resolve_ticket:
            reward, done = self._handle_resolve(expected)

        s.step_count      += 1
        s.accumulated_reward += reward

        # Timeout
        if s.step_count >= MAX_STEPS:
            done = True

        info["step_reward"]       = round(reward, 4)
        info["cumulative_reward"] = round(s.accumulated_reward, 4)

        return StepResult(
            observation = self._build_observation(),
            reward      = round(reward, 4),
            done        = done,
            info        = info,
        )

    # ─────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────

    def state(self) -> dict:
        """Return the full internal state as a dict (for debugging / grading)."""
        if self._state is None:
            return {}
        return self._state.model_dump()

    # ─────────────────────────────────────────────
    # grade()
    # ─────────────────────────────────────────────

    def grade(self) -> dict:
        """
        Run the official grader on the current episode and return score breakdown.
        Call this after the episode is done.
        """
        if self._task is None or self._state is None:
            raise RuntimeError("Call reset() and complete an episode first.")
        grader = get_grader(self._task["task_id"])
        return grader(self._state.model_dump(), self._task["expected"])

    # ─────────────────────────────────────────────
    # list_tasks()
    # ─────────────────────────────────────────────

    @staticmethod
    def list_tasks() -> list[dict]:
        return [
            {
                "task_id":     t["task_id"],
                "difficulty":  t["difficulty"],
                "description": t["description"],
            }
            for t in TASKS
        ]

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        return Observation(
            ticket_id        = s.ticket_id,
            customer_message = s.customer_message,
            customer_tier    = s.customer_tier,
            previous_tickets = s.previous_tickets,
            status           = s.status,
            assigned_team    = s.assigned_team,
            priority         = s.priority,
            category         = s.category,
            reply_draft      = s.reply_draft,
            step_count       = s.step_count,
            escalated        = s.escalated,
        )

    def _norm(self, val) -> str:
        """Normalise enum or string to plain string."""
        return val.value if hasattr(val, "value") else str(val)

    def _handle_classify(self, action: Action, expected: dict) -> float:
        expected_cat = self._norm(expected["category"])
        self._state.category = action.value
        self._state.status   = TicketStatus.in_progress
        if action.value == expected_cat:
            return +0.30
        return -0.10

    def _handle_priority(self, action: Action, expected: dict) -> float:
        expected_pri = self._norm(expected["priority"])
        self._state.priority = action.value
        if action.value == expected_pri:
            return +0.20
        return -0.05

    def _handle_assign(self, action: Action, expected: dict) -> float:
        expected_team = self._norm(expected["team"])
        self._state.assigned_team = action.value
        if action.value == expected_team:
            return +0.20
        return -0.10

    def _handle_reply(self, action: Action, expected: dict) -> float:
        self._state.reply_draft = action.value
        if not action.value.strip():
            return 0.0
        keywords = expected.get("reply_keywords", [])
        reply_lower = action.value.lower()
        hits = sum(1 for kw in keywords if kw in reply_lower)
        fraction = hits / max(len(keywords), 1)
        # Partial reward proportional to keyword coverage (max +0.20)
        return round(min(fraction * 2, 1.0) * 0.20, 4)

    def _handle_escalate(self, action: Action, expected: dict) -> float:
        self._state.escalated = True
        self._state.status    = TicketStatus.escalated
        if expected.get("should_escalate", False):
            return +0.10
        return -0.05   # unnecessary escalation

    def _handle_resolve(self, expected: dict) -> tuple[float, bool]:
        # Must have classified + assigned before resolving
        ready = (
            self._state.category is not None
            and self._state.assigned_team is not None
        )
        if not ready:
            return (-0.10, False)   # too early

        self._state.status = TicketStatus.resolved
        if expected.get("should_resolve", True) and not expected.get("should_escalate"):
            return (+0.10, True)
        # Resolving when escalation was expected is a mild error
        return (-0.05, True)
