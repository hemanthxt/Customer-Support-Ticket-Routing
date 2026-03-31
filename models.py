"""
models.py
Pydantic data models for TicketRoutingEnv.
Defines: Observation, Action, StepResult, TicketState
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class Category(str, Enum):
    billing  = "billing"
    technical = "technical"
    shipping  = "shipping"
    account   = "account"
    general   = "general"


class Priority(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"
    urgent = "urgent"


class Team(str, Enum):
    billing_team  = "billing_team"
    tech_team     = "tech_team"
    shipping_team = "shipping_team"
    account_team  = "account_team"
    support_team  = "support_team"


class ActionType(str, Enum):
    classify_ticket = "classify_ticket"
    set_priority    = "set_priority"
    assign_team     = "assign_team"
    draft_reply     = "draft_reply"
    escalate        = "escalate"
    resolve_ticket  = "resolve_ticket"


class TicketStatus(str, Enum):
    open       = "open"
    in_progress = "in_progress"
    escalated  = "escalated"
    resolved   = "resolved"


# ──────────────────────────────────────────────
# Observation  (what the agent sees)
# ──────────────────────────────────────────────

class Observation(BaseModel):
    ticket_id:        str
    customer_message: str
    customer_tier:    str                    # "standard" | "premium" | "enterprise"
    previous_tickets: int
    status:           TicketStatus
    assigned_team:    Optional[Team]  = None
    priority:         Optional[Priority] = None
    category:         Optional[Category] = None
    reply_draft:      str             = ""
    step_count:       int             = 0
    escalated:        bool            = False

    class Config:
        use_enum_values = True


# ──────────────────────────────────────────────
# Action  (what the agent does)
# ──────────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    value:       str = Field(..., description="Depends on action_type")

    class Config:
        use_enum_values = True


# ──────────────────────────────────────────────
# StepResult  (what step() returns)
# ──────────────────────────────────────────────

class StepResult(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        dict


# ──────────────────────────────────────────────
# Internal ticket state  (full truth)
# ──────────────────────────────────────────────

class TicketState(BaseModel):
    ticket_id:        str
    customer_message: str
    customer_tier:    str
    previous_tickets: int
    status:           TicketStatus = TicketStatus.open
    assigned_team:    Optional[Team]     = None
    priority:         Optional[Priority] = None
    category:         Optional[Category] = None
    reply_draft:      str            = ""
    step_count:       int            = 0
    escalated:        bool           = False
    actions_taken:    list[str]      = Field(default_factory=list)
    accumulated_reward: float        = 0.0

    class Config:
        use_enum_values = True
