import sqlite3
import re
from typing import Any, List, Optional

class MockConnection:
    def __init__(self, *args, **kwargs):
        # We always use an in-memory SQLite database to validate SQL syntax and types
        self.conn = sqlite3.connect(":memory:")
        self.last_cursor = None

    def execute(self, sql: str, params: Optional[List[Any]] = None):
        # Programmatically ignore CREATE SEQUENCE statements
        if re.search(r"CREATE\s+SEQUENCE", sql, re.IGNORECASE):
            self.last_cursor = None
            return self

        # Strip DEFAULT nextval('...') clauses so SQLite can auto-increment natively
        sql_clean = re.sub(
            r"DEFAULT\s+nextval\([^)]+\)",
            "",
            sql,
            flags=re.IGNORECASE
        )

        try:
            self.last_cursor = self.conn.execute(sql_clean, params or [])
            return self
        except sqlite3.Error as e:
            # Re-raise standard sqlite3 errors to fail tests on syntax bugs
            raise e

    def fetchone(self):
        if self.last_cursor is None:
            return None
        return self.last_cursor.fetchone()

    def fetchall(self):
        if self.last_cursor is None:
            return []
        return self.last_cursor.fetchall()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    @property
    def description(self):
        if self.last_cursor is None:
            return []
        return self.last_cursor.description

def connect(path=None):
    return MockConnection()
