# Testing Insights Agent

A conversational analytics agent that can:
- Generate read-only SQL from user questions
- Execute queries against your database (SQLAlchemy)
- Extract structured metrics from results
- Provide clear summaries + predictive recommendations for next tests
- Keep short-term memory (recent turns intact + older summary) and long-term memory (SQLite keyword/recency retrieval)

## Setup
1) Create a virtual env and install:
   pip install -r requirements.txt

2) Create a .env file (copy from .env.example) and fill values:
   - OPENAI_API_KEY
   - DATABASE_URL

3) Run:
   python -m testing_insights_agent.cli

## Notes
- This agent only executes SELECT queries (read-only).
- For best results, include the table name in your question.
