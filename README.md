# TicketRoutingEnv

> **A real-world OpenEnv simulation** where an AI agent processes customer
> support tickets by classifying issues, assigning priority, routing to the
> correct team, drafting replies, and resolving or escalating cases.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Running the Agent](#running-the-agent)
5. [Environment API](#environment-api)
6. [Observation Space](#observation-space)
7. [Action Space](#action-space)
8. [Reward System](#reward-system)
9. [Tasks](#tasks)
10. [Grading](#grading)
11. [Docker](#docker)
12. [Team](#team)

---

## Overview

`TicketRoutingEnv` simulates a business support workflow:

```
Customer sends ticket
      ↓
Agent reads observation
      ↓
Agent performs actions (classify → prioritize → assign → reply → resolve/escalate)
      ↓
Environment returns reward at each step
      ↓
Final grader returns score 0.0 – 1.0
```

The environment is:
- **deterministic** — same actions always produce the same rewards
- **partially rewarding** — partial credit for each correct sub-action
- **realistic** — mirrors actual support team workflows

---

## Project Structure

```
ticket-routing-env/
├── environment.py     # Core env: reset(), step(), state(), grade()
├── models.py          # Pydantic models: Observation, Action, StepResult, TicketState
├── tasks.py           # 3 benchmark tasks (easy / medium / hard)
├── graders.py         # Rule-based, deterministic graders
├── inference.py       # Baseline agent (heuristic + optional LLM)
├── openenv.yaml       # OpenEnv metadata and config
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
cd ticket-routing-env
pip install -r requirements.txt
```

### (Optional) LLM agent

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # default
```

---

## Running the Agent

```bash
# Run easy task with built-in heuristic agent
python inference.py --task TASK_01_EASY

# Run medium task
python inference.py --task TASK_02_MEDIUM

# Run hard task
python inference.py --task TASK_03_HARD

# Run all 3 tasks and print average score
python inference.py --all

# Use LLM agent (requires OPENAI_API_KEY)
python inference.py --all --llm
```

### Sample output

```
🎫 TicketRoutingEnv  |  Agent mode: Heuristic

============================================================
Task: TASK_01_EASY
Message: I was charged twice for my monthly subscription. Please fix this immediately.
Tier: standard  |  Prior tickets: 1
============================================================

[Step 1] Action → {"action_type": "classify_ticket", "value": "billing"}
         Reward  → +0.30  |  Cumulative: +0.30

[Step 2] Action → {"action_type": "set_priority", "value": "medium"}
         Reward  → +0.20  |  Cumulative: +0.50
...
────────────────────────────────────────────────────────────
Final grade breakdown:
  category         0.3000  ██████
  priority         0.2000  ████
  team             0.2000  ████
  reply            0.2000  ████
  handling         0.1000  ██
────────────────────────────────────────────────────────────
  TOTAL SCORE      1.0000
────────────────────────────────────────────────────────────
```

---

## Environment API

### Programmatic usage

```python
from environment import TicketRoutingEnv
from models import Action

env = TicketRoutingEnv()

# Start episode
obs = env.reset("TASK_01_EASY")
print(obs.model_dump())

# Take actions
result = env.step({"action_type": "classify_ticket", "value": "billing"})
print(result.reward, result.done)

# Get full internal state (debug)
print(env.state())

# Final score after episode ends
scores = env.grade()
print(scores)  # {"category": 0.30, "priority": ..., "total": 1.0}
```

---

## Observation Space

| Field              | Type    | Description                                 |
|--------------------|---------|---------------------------------------------|
| `ticket_id`        | string  | Unique ticket identifier                    |
| `customer_message` | string  | The raw customer complaint                  |
| `customer_tier`    | string  | `standard`, `premium`, or `enterprise`      |
| `previous_tickets` | integer | Number of prior complaints from customer    |
| `status`           | string  | `open`, `in_progress`, `escalated`, `resolved` |
| `assigned_team`    | string? | Which team is handling the ticket           |
| `priority`         | string? | Current priority level                      |
| `category`         | string? | Classified issue category                   |
| `reply_draft`      | string  | Current draft of the customer reply         |
| `step_count`       | integer | Number of actions taken so far              |
| `escalated`        | boolean | Whether ticket has been escalated           |

---

## Action Space

| `action_type`    | `value`                                                | Effect                        |
|------------------|--------------------------------------------------------|-------------------------------|
| `classify_ticket`| `billing` / `technical` / `shipping` / `account` / `general` | Sets category, marks in_progress |
| `set_priority`   | `low` / `medium` / `high` / `urgent`                  | Sets priority level           |
| `assign_team`    | `billing_team` / `tech_team` / `shipping_team` / `account_team` / `support_team` | Assigns team |
| `draft_reply`    | any text string                                        | Saves customer-facing reply   |
| `escalate`       | short reason string                                    | Marks ticket escalated        |
| `resolve_ticket` | `"done"`                                               | Closes ticket as resolved     |

---

## Reward System

| Event                            | Reward  |
|----------------------------------|---------|
| Correct category classification  | +0.30   |
| Wrong category                   | -0.10   |
| Correct priority                 | +0.20   |
| Wrong priority                   | -0.05   |
| Correct team assignment          | +0.20   |
| Wrong team                       | -0.10   |
| Useful reply (keyword coverage)  | up to +0.20 |
| Correct escalation               | +0.10   |
| Unnecessary escalation           | -0.05   |
| Correct resolution               | +0.10   |
| Resolving too early              | -0.10   |
| Repeated action type             | -0.05   |

**Maximum possible reward per episode: 1.00**

---

## Tasks

### TASK_01_EASY — Billing duplicate charge

> *"I was charged twice for my monthly subscription. Please fix this immediately."*

**Expected solution:**
- Category: `billing`
- Priority: `medium`
- Team: `billing_team`
- Reply must mention: refund / charge / billing
- Action: resolve

---

### TASK_02_MEDIUM — Urgent account lockout

> *"My account is locked and I need access before today's client meeting in two hours. This is extremely urgent!"*

**Expected solution:**
- Category: `account`
- Priority: `high`
- Team: `account_team`
- Reply must mention: urgent / access / unlock
- Action: resolve

---

### TASK_03_HARD — Multi-issue complaint

> *"My order has not arrived, I was still charged for it, and nobody replied to my last complaint three days ago."*

**Expected solution:**
- Category: `shipping` (primary)
- Priority: `urgent`
- Team: `shipping_team`
- Reply must address: order, charge, apology, follow-up
- Action: **escalate** (enterprise + ignored complaint)

---

## Grading

Each task is graded independently, returning a float in **[0.0, 1.0]**:

| Component  | Weight |
|------------|--------|
| Category   | 0.30   |
| Priority   | 0.20   |
| Team       | 0.20   |
| Reply      | 0.20   |
| Handling   | 0.10   |
| **Total**  | **1.00** |

Graders are **rule-based and deterministic** — no randomness.

---

## Docker

```bash
# Build
docker build -t ticket-routing-env .

# Run (all tasks, heuristic agent)
docker run --rm ticket-routing-env

# Run with LLM agent
docker run --rm -e OPENAI_API_KEY=sk-... ticket-routing-env \
    python inference.py --all --llm
```

---

## Team

| Member   | Responsibility                                               |
|----------|--------------------------------------------------------------|
| Member 1 | Core environment: `models.py`, `environment.py`              |
| Member 2 | Tasks & grading: `tasks.py`, `graders.py`                    |
| Member 3 | Deployment: `inference.py`, `Dockerfile`, HuggingFace Space  |

---

## One-line Summary

> *"TicketRoutingEnv is a real-world OpenEnv simulation where an AI agent processes customer support tickets by classifying issues, assigning priority, routing to the correct team, drafting replies, and resolving or escalating cases."*
