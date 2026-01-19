from __future__ import annotations
from typing import Any, Dict, List
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

class ReadOnlyDB:
    """
    Read-only SQL executor using SQLAlchemy.
    Enforces SELECT-only to prevent writes/drops/updates.
    """

    def __init__(self, database_url: str):
        self.engine: Engine = create_engine(database_url, future=True)

    @staticmethod
    def _is_safe_select(sql: str) -> bool:
        s = (sql or "").strip().lower()
        if not s.startswith("select"):
            return False
        banned = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "merge", "grant", "revoke"]
        return not any(re.search(rf"\b{b}\b", s) for b in banned)

    def run_sql(self, sql: str, limit_rows: int = 200) -> str:
        """
        Execute SQL and return a small markdown-like preview.
        """
        if not self._is_safe_select(sql):
            return f"SQL_ERROR: Unsafe or invalid SQL detected: {sql}"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchmany(limit_rows)
                cols = list(result.keys())

            # Render a simple markdown table
            if not cols:
                return "(no columns)"
            header = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            body_lines = []
            for r in rows:
                body_lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
            if not body_lines:
                body_lines.append("| " + " | ".join(["(no rows)"] + [""] * (len(cols) - 1)) + " |")
            return "\n".join([header, sep] + body_lines)

        except Exception as e:
            return f"SQL_ERROR: {str(e)}"
