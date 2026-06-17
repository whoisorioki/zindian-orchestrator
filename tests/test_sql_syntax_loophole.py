import pytest
import sqlite3
import duckdb


def test_sql_syntax_loophole_valid_queries():
    conn = duckdb.connect()
    # Test table creation with NEXTVAL stripping
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id INTEGER PRIMARY KEY DEFAULT nextval('test_seq'),
            name VARCHAR NOT NULL
        )
    """)
    # Test ignore CREATE SEQUENCE
    conn.execute("CREATE SEQUENCE test_seq")

    # Test valid insert and select
    conn.execute("INSERT INTO test_table (name) VALUES (?)", ["Alice"])
    conn.execute("INSERT INTO test_table (name) VALUES (?)", ["Bob"])

    cursor = conn.execute("SELECT * FROM test_table ORDER BY id ASC")
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0][1] == "Alice"
    assert rows[1][1] == "Bob"


def test_sql_syntax_loophole_catches_syntax_error():
    conn = duckdb.connect()
    # Invalid SQL syntax should raise a sqlite3 operational/programming error
    with pytest.raises(sqlite3.Error):
        conn.execute("CREATE TABLE_INVALID test_table (id INTEGER)")

    with pytest.raises(sqlite3.Error):
        conn.execute("INSERT INTO test_table VALUES (invalid_syntax)")
