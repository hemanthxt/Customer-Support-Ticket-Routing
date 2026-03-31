"""
tasks.py
Defines the 3 benchmark tasks for TicketRoutingEnv.
Each task is a dict with:
  - ticket        : initial ticket data
  - expected      : ground-truth solution used by the grader
  - difficulty    : "easy" | "medium" | "hard"
  - description   : human-readable label
"""

from models import Category, Priority, Team

TASKS: list[dict] = [

    # ──────────────────────────────────────────────
    # TASK 1 — Easy  (single-issue billing complaint)
    # ──────────────────────────────────────────────
    {
        "task_id":    "TASK_01_EASY",
        "difficulty": "easy",
        "description": "Simple duplicate-charge billing complaint",

        "ticket": {
            "ticket_id":        "T101",
            "customer_message": (
                "I was charged twice for my monthly subscription. "
                "Please fix this immediately."
            ),
            "customer_tier":    "standard",
            "previous_tickets": 1,
        },

        "expected": {
            "category":  Category.billing,
            "priority":  Priority.medium,
            "team":      Team.billing_team,
            # reply must mention at least one of these keywords
            "reply_keywords": ["refund", "charge", "billing", "payment", "duplicate"],
            # should NOT escalate; should resolve or mark for follow-up
            "should_escalate": False,
            "should_resolve":  True,
        },
    },

    # ──────────────────────────────────────────────
    # TASK 2 — Medium  (urgent account-access issue)
    # ──────────────────────────────────────────────
    {
        "task_id":    "TASK_02_MEDIUM",
        "difficulty": "medium",
        "description": "Urgent account lockout before a client meeting",

        "ticket": {
            "ticket_id":        "T202",
            "customer_message": (
                "My account is locked and I need access before today's client "
                "meeting in two hours. This is extremely urgent!"
            ),
            "customer_tier":    "premium",
            "previous_tickets": 0,
        },

        "expected": {
            "category":  Category.account,
            "priority":  Priority.high,
            "team":      Team.account_team,
            "reply_keywords": ["urgent", "access", "account", "unlock", "priority"],
            "should_escalate": False,
            "should_resolve":  True,
        },
    },

    # ──────────────────────────────────────────────
    # TASK 3 — Hard  (multi-issue: shipping + billing + ignored complaint)
    # ──────────────────────────────────────────────
    {
        "task_id":    "TASK_03_HARD",
        "difficulty": "hard",
        "description": "Multi-issue: missing order, double charge, no prior response",

        "ticket": {
            "ticket_id":        "T303",
            "customer_message": (
                "My order has not arrived, I was still charged for it, "
                "and nobody replied to my last complaint three days ago. "
                "This is completely unacceptable."
            ),
            "customer_tier":    "enterprise",
            "previous_tickets": 3,
        },

        "expected": {
            # Primary category for routing; grader checks both shipping + billing keywords in reply
            "category":  Category.shipping,
            "priority":  Priority.urgent,
            "team":      Team.shipping_team,
            "reply_keywords": [
                "order", "delivery", "charge", "refund", "apologize",
                "escalate", "follow", "complaint",
            ],
            # High frustration + ignored complaint → escalation expected
            "should_escalate": True,
            "should_resolve":  False,   # escalation takes priority over direct resolve
        },
    },
]


def get_task(task_id: str) -> dict:
    """Return a task by its task_id string."""
    for t in TASKS:
        if t["task_id"] == task_id:
            return t
    raise ValueError(f"Unknown task_id: {task_id!r}")


def get_all_tasks() -> list[dict]:
    return TASKS
