"""
test_env.py — smoke-test all 3 tasks end-to-end.
Run with:  python test_env.py
"""
import json
import traceback
from environment import TicketRoutingEnv

env = TicketRoutingEnv()
errors = []

TASK_IDS = ["TASK_01_EASY", "TASK_02_MEDIUM", "TASK_03_HARD"]

# A generic action sequence to exercise every code path
ACTION_SEQUENCES = {
    "TASK_01_EASY": [
        {"action_type": "classify_ticket", "value": "billing"},
        {"action_type": "set_priority",    "value": "medium"},
        {"action_type": "assign_team",     "value": "billing_team"},
        {"action_type": "draft_reply",     "value": "We will process your refund for the duplicate charge."},
        {"action_type": "resolve_ticket",  "value": "done"},
    ],
    "TASK_02_MEDIUM": [
        {"action_type": "classify_ticket", "value": "account"},
        {"action_type": "set_priority",    "value": "high"},
        {"action_type": "assign_team",     "value": "account_team"},
        {"action_type": "draft_reply",     "value": "We understand this is urgent. Our team will unlock your account immediately."},
        {"action_type": "resolve_ticket",  "value": "done"},
    ],
    "TASK_03_HARD": [
        {"action_type": "classify_ticket", "value": "shipping"},
        {"action_type": "set_priority",    "value": "urgent"},
        {"action_type": "assign_team",     "value": "shipping_team"},
        {"action_type": "draft_reply",     "value": "We apologise for the missing order and the charge. We will follow up on your complaint and escalate immediately."},
        {"action_type": "escalate",        "value": "Enterprise customer with repeated unresolved complaints"},
    ],
}

for task_id in TASK_IDS:
    try:
        obs = env.reset(task_id)
        assert obs.ticket_id,        "ticket_id missing"
        assert obs.customer_message, "customer_message missing"
        assert obs.status == "open", f"Expected open, got {obs.status}"

        actions = ACTION_SEQUENCES[task_id]
        for step_idx, action_dict in enumerate(actions):
            result = env.step(action_dict)
            assert isinstance(result.reward, float), f"reward must be float, got {type(result.reward)}"
            assert isinstance(result.done,   bool),  f"done must be bool, got {type(result.done)}"
            if result.done:
                break

        scores = env.grade()
        assert "total" in scores, "grade() must return 'total' key"
        assert 0.0 <= scores["total"] <= 1.0, f"score out of range: {scores['total']}"

        breakdown = {k: v for k, v in scores.items() if k != "total"}
        print(f"OK  {task_id:<20}  total={scores['total']:.4f}  {json.dumps(breakdown)}")

    except Exception as exc:
        errors.append((task_id, str(exc)))
        traceback.print_exc()

print()
if errors:
    print("FAILURES:")
    for tid, msg in errors:
        print(f"  {tid}: {msg}")
    raise SystemExit(1)
else:
    print("ALL TASKS PASS — no errors found.")
