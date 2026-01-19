from __future__ import annotations
from typing import TypedDict, Dict, List, Tuple
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .prompts import (
    SQL_GEN_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    METRIC_EXTRACTION_PROMPT,
    PREDICTIVE_SYSTEM_PROMPT,
)
from .memory import update_short_term_memory, LongTermMemorySQLite

class AppState(TypedDict, total=False):
    user_id: int
    user_query: str
    memory: Dict[str, object]      # {older_summary: str, recent_turns: [{role,content}, ...]}
    metrics: Dict[str, float]
    sql: str
    sql_results: str
    answer: str

def _needs_sql(user_q: str) -> bool:
    q = (user_q or "").lower()
    triggers = ["show", "give", "find", "compare", "insight", "trend", "top", "count", "sum", "percent", "table", "rows", "columns"]
    return any(t in q for t in triggers) or bool(re.search(r"\bfrom\b|\bselect\b", q))

def _is_predictive(user_q: str) -> bool:
    q = (user_q or "").lower()
    triggers = ["suggest", "recommend", "next test", "predict", "what should we do", "how to improve", "strategy", "proposal"]
    return any(t in q for t in triggers)

def build_graph(chat, db, lt_store: LongTermMemorySQLite):
    """
    Returns a compiled LangGraph app with:
      - handle_query (routes SQL vs non-SQL + uses long/short memory)
      - optional metric extraction + predictive recommendations
    """

    def handle_query(state: AppState) -> AppState:
        user_id = state.get("user_id")
        user_q = state.get("user_query", "")
        memory = state.get("memory") or {"older_summary": "", "recent_turns": []}
        metrics = state.get("metrics") or {}

        # ----- Long-term retrieval -----
        lt_hits: List[Tuple[float, str]] = []
        if user_id is not None:
            lt_hits = lt_store.search(user_id=user_id, query=user_q, k=5, min_score=0.15)

        # ----- Build context from short-term memory (KEEP IT INTACT) -----
        context_msgs: List[HumanMessage] = []
        if lt_hits:
            mem_text = "\n".join([f"- (score={s:.3f}) {t}" for s, t in lt_hits])
            context_msgs.append(HumanMessage(content=f"Relevant long-term memories:\n{mem_text}"))
        if memory.get("older_summary"):
            context_msgs.append(HumanMessage(content=f"Earlier summary:\n{memory['older_summary']}"))
        for turn in memory.get("recent_turns", []):
            # preserve roles by prefixing
            context_msgs.append(HumanMessage(content=f"{turn['role'].upper()}: {turn['content']}"))

        # ----- If predictive question: produce recommendations (no SQL required) -----
        if _is_predictive(user_q):
            prompt = [
                SystemMessage(content=PREDICTIVE_SYSTEM_PROMPT),
                *context_msgs,
                HumanMessage(content=f"User question:\n{user_q}\n\nKnown metrics:\n{metrics}")
            ]
            resp = chat.invoke(prompt).content.strip()

            memory = update_short_term_memory(memory, user_q, resp)
            if user_id is not None:
                lt_store.add_memory(user_id, f"User: {user_q}\nAssistant: {resp}")
            return {"answer": resp, "memory": memory, "metrics": metrics}

        # ----- SQL path if needed -----
        if _needs_sql(user_q):
            # IMPORTANT FIX vs your failing version:
            # We do NOT invent columns like 'date'. We instruct the model to either:
            # - query the mentioned table directly with SELECT * LIMIT 20 (safe)
            # - or use information_schema if necessary.
            sql_prompt = [
                SystemMessage(content=SQL_GEN_SYSTEM_PROMPT),
                *context_msgs,
                HumanMessage(content=f"User question:\n{user_q}\n\nReturn ONLY a SELECT SQL query.")
            ]
            sql = chat.invoke(sql_prompt).content.strip()

            sql_results = db.run_sql(sql, limit_rows=200)

            # Metric extraction (best-effort)
            metric_prompt = [
                SystemMessage(content=METRIC_EXTRACTION_PROMPT),
                HumanMessage(content=f"SQL results (markdown):\n{sql_results}")
            ]
            metric_text = chat.invoke(metric_prompt).content.strip()
            try:
                parsed = json.loads(metric_text) if metric_text else {}
                for k, v in (parsed or {}).items():
                    try:
                        metrics[str(k)] = float(v)
                    except Exception:
                        pass
            except Exception:
                pass

            # Explanation
            qa_prompt = [
                SystemMessage(content=QA_SYSTEM_PROMPT),
                *context_msgs,
                HumanMessage(content=f"""
User question:
{user_q}

SQL executed:
{sql}

SQL results:
{sql_results}

Known structured metrics:
{metrics}
""")
            ]
            answer = chat.invoke(qa_prompt).content.strip()

            memory = update_short_term_memory(memory, user_q, answer)
            if user_id is not None:
                lt_store.add_memory(user_id, f"User: {user_q}\nSQL: {sql}\nResult: {sql_results}\nAssistant: {answer}")
            return {"answer": answer, "memory": memory, "metrics": metrics, "sql": sql, "sql_results": sql_results}

        # ----- Non-SQL QA path -----
        qa_prompt = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            *context_msgs,
            HumanMessage(content=f"User question:\n{user_q}\n\nKnown structured metrics:\n{metrics}")
        ]
        answer = chat.invoke(qa_prompt).content.strip()

        memory = update_short_term_memory(memory, user_q, answer)
        if user_id is not None:
            lt_store.add_memory(user_id, f"User: {user_q}\nAssistant: {answer}")
        return {"answer": answer, "memory": memory, "metrics": metrics}

    graph = StateGraph(AppState)
    graph.add_node("handle_query", handle_query)
    graph.add_edge(START, "handle_query")
    graph.add_edge("handle_query", END)
    return graph.compile()
