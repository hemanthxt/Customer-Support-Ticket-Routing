# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="Team TicketRouting"
LABEL description="TicketRoutingEnv — OpenEnv customer-support ticket routing simulation"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default: run all 3 tasks with the heuristic agent
CMD ["python", "inference.py", "--all"]
