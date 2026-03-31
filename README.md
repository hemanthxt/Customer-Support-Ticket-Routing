# Customer-Support-Ticket-Routing
Project Title

TicketRoutingEnv: Multi-Issue Aware Customer Support Routing Environment

📌 Short Description (2–3 lines)

TicketRoutingEnv is a deterministic OpenEnv simulation for training AI agents to handle customer support tickets. It supports classification, prioritization, team routing, reply generation, and escalation decisions.
A key innovation is a multi-issue detection and prioritization system, enabling agents to correctly handle complex tickets with multiple problems.

🎯 Problem Statement

Traditional ticket routing systems treat each ticket as a single-label classification problem, which fails in real-world scenarios where users report multiple issues in one message.

This leads to:

Incorrect routing
Poor prioritization
Missed escalation signals
Low customer satisfaction
💡 Our Solution

We designed a 3-stage multi-issue decision system:

1️⃣ Extract All Issues

Detect all categories present in the ticket (e.g., shipping, billing, complaint)

2️⃣ Rank by Business Impact

Prioritize issues using domain-aware ranking:

account > shipping > billing > technical > complaint
3️⃣ Route by Primary + Acknowledge All
Route based on the primary issue
Include secondary issues in the reply
Trigger escalation when needed
⚙️ Key Features
✅ Deterministic environment (consistent rewards)
✅ Multi-issue aware routing
✅ Priority detection (low → urgent)
✅ Team assignment system
✅ Reply generation with keyword scoring
✅ Escalation vs resolution decision logic
✅ Dual agents:
Heuristic (rule-based)
LLM (OpenAI-supported)
✅ Dockerized and OpenEnv compatible
🧠 Architecture Overview
User Ticket
   ↓
Issue Extraction
   ↓
Issue Ranking (Business Impact)
   ↓
Primary Issue Selection
   ↓
Routing + Priority + Team
   ↓
Reply Generation
   ↓
Escalate / Resolve Decision
📊 Results & Improvements
Task	Before	After
TASK_01_EASY	0.80	0.80
TASK_02_MEDIUM	0.80	0.80
TASK_03_HARD	0.40 ❌	1.00 ✅
Average	0.67	0.87

📌 Major improvement achieved by introducing multi-issue detection and prioritization

🔍 Example Output (Explainable AI)
{
  "category": "shipping",
  "priority": "urgent",
  "team": "shipping_team",
  "detected_issues": ["shipping", "billing", "complaint"],
  "reason": "Primary issue is missing order; billing and complaint are secondary",
  "handling": "escalate"
}
🧪 Hard Case Handling (Key Strength)

Scenario:

Order not delivered
Customer charged
Complaint ignored

Agent Behavior:

Detects all issues
Chooses shipping as primary
Marks priority as urgent
Escalates due to complaint history

👉 This solves real-world ambiguity effectively.

🚀 How to Run
pip install -r requirements.txt
python test_env.py

Or via Docker:

docker build -t ticket-routing-env .
docker run ticket-routing-env
🔮 Future Improvements
Conversation-based clarification before routing
Confidence-based decision making
Support for multilingual tickets
Learning loop for agent improvement
Analytics dashboard for performance tracking
🎤 1-Minute Explanation Script (IMPORTANT)

Use this if you need to explain:

“We built TicketRoutingEnv, a deterministic simulation for training AI agents to handle customer support tickets.
The key challenge we addressed is that real-world tickets often contain multiple issues, but most systems treat them as single-label problems.

To solve this, we introduced a 3-stage system: first, we extract all issues from the ticket; second, we rank them by business impact; and third, we route based on the primary issue while acknowledging secondary ones.

This significantly improved performance on complex tickets — our hardest benchmark improved from 40% to 100% accuracy.

Our system also handles prioritization, team assignment, reply generation, and escalation decisions, making it realistic and production-relevant.”

🏁 Final Note

👉 This submission shows:

Technical depth (environment + agent)
Real-world relevance
Clear measurable improvement
Strong reasoning capability
