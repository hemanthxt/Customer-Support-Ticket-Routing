"""
inference.py
Baseline AI agent for TicketRoutingEnv.

The agent uses the OpenAI Chat Completions API (compatible with any
OpenAI-compatible endpoint) to decide which action to take at each step.

Usage:
    python inference.py --task TASK_01_EASY
    python inference.py --task TASK_02_MEDIUM
    python inference.py --task TASK_03_HARD
    python inference.py --all
"""

from __future__ import annotations
import argparse
import json
import os
import sys

# Force UTF-8 output on Windows terminals (fixes UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── guard: openai optional so env works without it ──
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from environment import TicketRoutingEnv



# ──────────────────────────────────────────────────────────────
# System prompt for the LLM agent
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert AI customer-support agent.
You receive a JSON observation of a support ticket and must respond with
a single JSON action to take.

Valid action_types and their value formats:
  classify_ticket  → value: one of [billing, technical, shipping, account, general]
  set_priority     → value: one of [low, medium, high, urgent]
  assign_team      → value: one of [billing_team, tech_team, shipping_team, account_team, support_team]
  draft_reply      → value: a concise, empathetic reply string (max 3 sentences)
  escalate         → value: short reason string
  resolve_ticket   → value: "done"

Rules:
1. Always classify_ticket first.
2. Then set_priority.
3. Then assign_team.
4. Then draft_reply.
5. Then escalate (only if genuinely needed) OR resolve_ticket.
6. Never repeat the same action_type.
7. Output ONLY valid JSON — no markdown, no explanation.

Example output:
{"action_type": "classify_ticket", "value": "billing"}
""".strip()


# ──────────────────────────────────────────────────────────────
# LLM-based agent
# ──────────────────────────────────────────────────────────────

def llm_agent_step(client: "OpenAI", obs_dict: dict) -> dict:
    """Ask the LLM what action to take given the current observation."""
    response = client.chat.completions.create(
        model    = os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": json.dumps(obs_dict, indent=2)},
        ],
        temperature    = 0,
        max_tokens     = 200,
        response_format= {"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ──────────────────────────────────────────────────────────────
# Multi-issue detection and priority ranking
# ──────────────────────────────────────────────────────────────

# Business impact priority for each issue category
ISSUE_PRIORITY = {
    "account":    100,      # Account access is critical
    "shipping":   80,       # Product/order fulfillment
    "billing":    60,       # Financial concerns
    "technical":  50,       # System functionality
    "complaint":  40,       # Service quality / responsiveness
    "general":    10,
}

# Keywords for each issue type
ISSUE_KEYWORDS = {
    "account":    ["account", "locked", "login", "access", "password", "username", "unlock"],
    "shipping":   ["order", "delivery", "ship", "arrive", "arrival", "package", "item", "product"],
    "billing":    ["charge", "charged", "billed", "payment", "refund", "invoice", "subscription"],
    "technical":  ["error", "bug", "crash", "broken", "not working", "slow", "lag"],
    "complaint":  ["complaint", "complained", "nobody replied", "no response", "ignored", "unacceptable", "disappointed"],
}


def extract_all_issues(msg: str) -> list[str]:
    """
    Detect ALL issue categories mentioned in the message.
    Returns list of issue types sorted by priority (highest first).
    """
    msg_lower = msg.lower()
    detected = []
    
    for issue_type, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in msg_lower:
                detected.append(issue_type)
                break  # Only count each issue type once
    
    # Sort by business priority (highest first)
    detected.sort(key=lambda x: ISSUE_PRIORITY.get(x, 0), reverse=True)
    return detected


def determine_primary_issue(msg: str) -> tuple[str, list[str]]:
    """
    Determine primary issue and list of secondary issues.
    Returns (primary_issue, all_issues_list)
    """
    all_issues = extract_all_issues(msg)
    if not all_issues:
        return "general", []
    primary = all_issues[0]
    secondary = all_issues[1:] if len(all_issues) > 1 else []
    return primary, all_issues


def should_escalate(msg: str, obs: dict, primary_issue: str, all_issues: list[str]) -> bool:
    """
    Decide if ticket should be escalated based on business rules.
    """
    msg_lower = msg.lower()
    
    # Escalate if multiple serious issues
    if len(all_issues) >= 2:
        if "complaint" in all_issues or "billing" in all_issues:
            return True
    
    # Escalate if ignored complaint + other issues
    if "complaint" in all_issues and len(all_issues) >= 2:
        return True
    
    # Escalate for enterprise tier with prior complaints/issues
    if obs.get("customer_tier") == "enterprise" and obs.get("previous_tickets", 0) >= 3:
        return True
    
    # Escalate if customer mentions being ignored
    if any(w in msg_lower for w in ["nobody replied", "no response", "ignored"]):
        return True
    
    return False


def determine_priority(msg: str, obs: dict, all_issues: list[str]) -> str:
    """
    Determine priority level based on urgency and issue complexity.
    """
    msg_lower = msg.lower()
    
    # Urgent signals
    urgent_words = ["urgent", "immediately", "critical", "unacceptable", "emergency"]
    if any(w in msg_lower for w in urgent_words):
        return "urgent"
    
    # Check for time pressure
    if any(w in msg_lower for w in ["meeting", "today", "now", "asap", "within hours"]):
        return "high"
    
    # Multiple issues warrant higher priority
    if len(all_issues) >= 2:
        return "high"
    
    # Enterprise customer + prior issues
    if obs.get("customer_tier") in ["premium", "enterprise"] and obs.get("previous_tickets", 0) >= 2:
        return "high"
    
    # Financial harm (billing + shipping)
    if "billing" in all_issues and "shipping" in all_issues:
        return "high"
    
    # Default
    if obs.get("customer_tier") in ["premium", "enterprise"]:
        return "medium"
    
    return "medium"


# ──────────────────────────────────────────────────────────────
# Heuristic fallback agent with multi-issue support
# ──────────────────────────────────────────────────────────────

def heuristic_agent_step(obs: dict, step_index: int) -> dict:
    """
    Enhanced rule-based agent with multi-issue detection.
    
    Workflow:
    1. Extract primary and secondary issues from message
    2. Use primary issue for classification and team assignment
    3. Set priority based on issue complexity and urgency
    4. Draft reply mentioning all relevant issues
    5. Escalate if appropriate, else resolve
    """
    msg = obs["customer_message"]
    msg_lower = msg.lower()
    
    # Determine all issues once and cache for this episode
    primary_issue, all_issues = determine_primary_issue(msg)
    
    # Classification: use primary issue
    if step_index == 0:
        category_map = {
            "account":   "account",
            "shipping":  "shipping",
            "billing":   "billing",
            "technical": "technical",
            "complaint": "general",
            "general":   "general",
        }
        primary = category_map.get(primary_issue, "general")
        return {"action_type": "classify_ticket", "value": primary}
    
    if step_index == 1:
        # Determine priority based on complexity and urgency
        priority = determine_priority(msg, obs, all_issues)
        return {"action_type": "set_priority", "value": priority}
    
    if step_index == 2:
        # Team assignment from primary issue
        category = obs.get("category", "general")
        team_map = {
            "billing":   "billing_team",
            "account":   "account_team",
            "shipping":  "shipping_team",
            "technical": "tech_team",
            "general":   "support_team",
        }
        return {"action_type": "assign_team", "value": team_map.get(category, "support_team")}
    
    if step_index == 3:
        # Draft reply mentioning all detected issues
        category = obs.get("category", "general")
        tier = obs.get("customer_tier", "standard")
        
        # Build reply acknowledging multiple issues
        acknowledgments = []
        if "shipping" in all_issues:
            acknowledgments.append("your order not arriving")
        if "billing" in all_issues:
            acknowledgments.append("the unexpected charge")
        if "complaint" in all_issues:
            acknowledgments.append("your previous complaint not being addressed")
        if "account" in all_issues:
            acknowledgments.append("your account access being locked")
        if "technical" in all_issues:
            acknowledgments.append("the technical issues you're experiencing")
        
        issue_text = " and ".join(acknowledgments) if acknowledgments else "your concern"
        
        # Multi-issue aware replies
        if len(all_issues) >= 2:
            reply = (
                f"We sincerely apologize for {issue_text}. This is clearly unacceptable, "
                f"especially that your previous concern went unanswered. We are treating this as high-priority "
                f"and investigating all aspects immediately."
            )
        elif category == "billing":
            reply = (
                f"We sincerely apologize for the charges on your account. Our billing team will "
                f"review and process a full refund within 2–3 business days."
            )
        elif category == "account":
            reply = (
                "We understand this is urgent and we are prioritizing your account unlock. "
                "Our account team will restore access immediately."
            )
        elif category == "shipping":
            reply = (
                "We are very sorry your order has not arrived. Our shipping team is investigating "
                "and will resolve this for you right away."
            )
        else:
            reply = (
                "Thank you for bringing this to our attention. We are investigating and will respond shortly."
            )
        
        return {"action_type": "draft_reply", "value": reply}
    
    if step_index == 4:
        # Decide escalate vs resolve
        escalate = should_escalate(msg, obs, primary_issue, all_issues)
        
        if escalate:
            reason = "Multiple unresolved issues and ignored complaint—escalating to management"
            return {"action_type": "escalate", "value": reason}
        else:
            return {"action_type": "resolve_ticket", "value": "done"}
    
    # Default: resolve
    return {"action_type": "resolve_ticket", "value": "done"}


# ──────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool, client=None) -> dict:
    env = TicketRoutingEnv()
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"Message: {obs.customer_message}")
    print(f"Tier: {obs.customer_tier}  |  Prior tickets: {obs.previous_tickets}")
    print(f"{'='*60}")

    done       = False
    step_index = 0


    while not done:
        if use_llm and client:
            action_dict = llm_agent_step(client, obs_dict)
        else:
            action_dict = heuristic_agent_step(obs_dict, step_index)

        print(f"\n[Step {step_index+1}] Action -> {json.dumps(action_dict)}")

        result   = env.step(action_dict)
        obs_dict = result.observation.model_dump()
        # track step count via obs_dict (total_rwd not needed here)

        print(f"         Reward  -> {result.reward:+.2f}  |  Cumulative: {result.info.get('cumulative_reward', 0):+.2f}")
        if result.info.get("penalty"):
            print(f"         [!] Penalty: {result.info['penalty']}")

        done       = result.done
        step_index += 1

    sep = "-" * 60
    print(f"\n{sep}")
    scores = env.grade()
    print("Final grade breakdown:")
    for k, v in scores.items():
        label = f"  {k:<16}"
        bar   = "#" * int(v * 20)
        print(f"{label} {v:.4f}  {bar}")
    print(sep)
    print(f"  TOTAL SCORE      {scores['total']:.4f}")
    print(f"{sep}\n")

    return scores


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TicketRoutingEnv baseline agent runner"
    )
    parser.add_argument(
        "--task", default="TASK_01_EASY",
        choices=["TASK_01_EASY", "TASK_02_MEDIUM", "TASK_03_HARD"],
        help="Task ID to run",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three tasks sequentially",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use OpenAI LLM agent (requires OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    # Set up client if LLM mode requested
    client = None
    use_llm = args.llm and OPENAI_AVAILABLE
    if use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set. Falling back to heuristic agent.")
            use_llm = False
        else:
            client = OpenAI(api_key=api_key)
    elif args.llm and not OPENAI_AVAILABLE:
        print("WARNING: openai package not installed. Falling back to heuristic agent.")

    mode = "LLM" if use_llm else "Heuristic"
    print(f"\n[TicketRoutingEnv]  Agent mode: {mode}")

    tasks = ["TASK_01_EASY", "TASK_02_MEDIUM", "TASK_03_HARD"] if args.all else [args.task]

    all_scores = {}
    for task_id in tasks:
        scores = run_episode(task_id, use_llm, client)
        all_scores[task_id] = scores["total"]

    if args.all:
        avg = sum(all_scores.values()) / len(all_scores)
        print("\nSummary across all tasks:")
        for tid, sc in all_scores.items():
            print(f"   {tid}: {sc:.4f}")
        print(f"   AVERAGE:        {avg:.4f}")


if __name__ == "__main__":
    main()
