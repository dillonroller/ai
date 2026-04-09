import sqlite3
from pathlib import Path
from contextlib import closing

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "library.db"

mcp = FastMCP("library")


class Book(BaseModel):
    id: int
    title: str
    author: str
    year: int
    rating: float


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    if DB_PATH.exists():
        return
    with closing(connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                year INTEGER NOT NULL,
                rating REAL NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO books (title, author, year, rating) VALUES (?, ?, ?, ?)",
            [
                ("The Pragmatic Programmer", "Andrew Hunt", 1999, 4.6),
                ("Designing Data-Intensive Applications", "Martin Kleppmann", 2017, 4.8),
                ("The Rust Programming Language", "Steve Klabnik", 2019, 4.7),
                ("Fluent Python", "Luciano Ramalho", 2015, 4.5),
                ("Crafting Interpreters", "Robert Nystrom", 2021, 4.9),
            ],
        )


def row_to_book(row: sqlite3.Row) -> Book:
    return Book(**dict(row))


@mcp.tool()
def search_books(query: str, min_rating: float = 0.0) -> list[Book]:
    """Full-text search across title and author, optionally filtered by rating."""
    sql = """
        SELECT * FROM books
        WHERE (title LIKE ? OR author LIKE ?) AND rating >= ?
        ORDER BY rating DESC
    """
    like = f"%{query}%"
    with closing(connect()) as conn:
        rows = conn.execute(sql, (like, like, min_rating)).fetchall()
    return [row_to_book(r) for r in rows]


@mcp.tool()
def add_book(title: str, author: str, year: int, rating: float) -> Book:
    """Add a book to the library and return the inserted row."""
    with closing(connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO books (title, author, year, rating) VALUES (?, ?, ?, ?) RETURNING *",
            (title, author, year, rating),
        )
        return row_to_book(cur.fetchone())


@mcp.tool()
def top_books(limit: int = 5) -> list[Book]:
    """Return the highest-rated books."""
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM books ORDER BY rating DESC LIMIT ?", (limit,)
        ).fetchall()
    return [row_to_book(r) for r in rows]


@mcp.resource("library://schema")
def schema() -> str:
    """Expose the database schema as a readable resource."""
    with closing(connect()) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    return "\n\n".join(r["sql"] for r in rows)


@mcp.prompt()
def recommend(mood: str) -> str:
    """Generate a recommendation prompt seeded by mood."""
    return (
        f"Use the `top_books` and `search_books` tools to recommend a book "
        f"for someone who is feeling {mood}. Explain your reasoning in two sentences."
    )


def main() -> None:
    init_db()
    mcp.run()


if __name__ == "__main__":
    main()
