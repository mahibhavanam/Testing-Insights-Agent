from __future__ import annotations
from getpass import getpass

from .config import get_settings
from .llm import OpenAIChat
from .db import ReadOnlyDB
from .memory import LongTermMemorySQLite
from .graph import build_graph

def main():
    s = get_settings()

    chat = OpenAIChat(
        api_key=s.openai_api_key,
        model=s.openai_model,
        temperature=0.0,
        max_tokens=2000,
    )
    db = ReadOnlyDB(s.database_url)
    lt_store = LongTermMemorySQLite(s.long_term_db_path)

    # login
    print("Testing Insights Agent")
    username = input("Username: ").strip()
    password = getpass("Password (stored hashed locally in SQLite): ").strip()
    user_id = lt_store.create_or_auth(username, password)

    compiled = build_graph(chat, db, lt_store)

    # persistent session state
    state = {
        "user_id": user_id,
        "memory": {"older_summary": "", "recent_turns": []},
        "metrics": {},
    }

    while True:
        q = input("\nAsk your query (exit/quit/q): ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("\nSession ended.")
            break

        state = compiled.invoke({
            "user_id": state["user_id"],
            "user_query": q,
            "memory": state["memory"],
            "metrics": state["metrics"],
        })

        print("\n" + state["answer"])

if __name__ == "__main__":
    main()
