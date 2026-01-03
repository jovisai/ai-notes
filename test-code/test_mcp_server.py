"""Test MCP server for article - connects to a SQLite database."""
from typing import Any
from mcp.server.fastmcp import FastMCP
import sqlite3
import json

# Initialize FastMCP server
mcp = FastMCP("database-query")

# Database path
DB_PATH = "test_data.db"


def init_db():
    """Initialize test database with sample data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Insert sample data
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            [
                ("Alice Johnson", "alice@example.com"),
                ("Bob Smith", "bob@example.com"),
                ("Carol White", "carol@example.com"),
            ]
        )
        cursor.executemany(
            "INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)",
            [
                (1, "Widget A", 29.99),
                (1, "Widget B", 49.99),
                (2, "Gadget X", 199.99),
                (3, "Widget A", 29.99),
                (3, "Service Plan", 99.99),
            ]
        )

    conn.commit()
    conn.close()


@mcp.tool()
def list_tables() -> str:
    """List all tables in the database.

    Returns a list of table names available for querying.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    conn.close()
    return json.dumps({"tables": tables}, indent=2)


@mcp.tool()
def describe_table(table_name: str) -> str:
    """Get the schema of a specific table.

    Args:
        table_name: Name of the table to describe
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    if not columns:
        conn.close()
        return f"Table '{table_name}' not found."

    schema = [
        {"name": col[1], "type": col[2], "nullable": not col[3], "primary_key": bool(col[5])}
        for col in columns
    ]

    conn.close()
    return json.dumps({"table": table_name, "columns": schema}, indent=2)


@mcp.tool()
def run_query(sql: str) -> str:
    """Execute a read-only SQL query on the database.

    Args:
        sql: The SQL query to execute (SELECT statements only)
    """
    # Security: Only allow SELECT statements
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

        results = [dict(zip(columns, row)) for row in rows]

        conn.close()
        return json.dumps({"columns": columns, "rows": results, "count": len(results)}, indent=2)
    except sqlite3.Error as e:
        conn.close()
        return f"SQL Error: {str(e)}"


# Initialize database on import
init_db()


if __name__ == "__main__":
    # For testing: run a simple query
    print("Testing MCP server tools...")
    print("\n1. List tables:")
    print(list_tables())
    print("\n2. Describe users table:")
    print(describe_table("users"))
    print("\n3. Run query:")
    print(run_query("SELECT u.name, o.product, o.amount FROM users u JOIN orders o ON u.id = o.user_id"))
