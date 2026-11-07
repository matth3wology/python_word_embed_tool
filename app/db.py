from pgvector import Vector
import tomllib  # built in on Python 3.11+
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

# CONFIG_PATH = Path(__file__).parent / "config.toml"
CONFIG_PATH = "../config.toml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:  # tomllib requires binary mode
        return tomllib.load(f)


config = load_config()
DIM = config["embedding"]["dim"]


def get_connection() -> psycopg.Connection:
    """Return a ready-to-use connection with pgvector registered."""
    db = config["database"]
    conn = psycopg.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"],
        autocommit=True,
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db(conn: psycopg.Connection) -> None:
    """Create the table if it doesn't exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS items (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({DIM})
        )
    """)


def insert_item(conn: psycopg.Connection, content: str, embedding) -> int:
    """Insert text and its embedding, return the new row id."""
    row = conn.execute(
        "INSERT INTO items (content, embedding) VALUES (%s, %s) RETURNING id",
        (content, Vector(embedding)),
    ).fetchone()
    return row[0]


def insert_items(conn: psycopg.Connection, content) -> int:
    """Load chunks into the DB"""
    for chunk, vec in content:
        insert_item(conn, chunk, vec)


def search(conn: psycopg.Connection, embedding, limit: int = 5):
    """Return [(content, similarity), ...] nearest first."""
    vec = Vector(embedding)
    return conn.execute(
        "SELECT content, 1 - (embedding <=> %s) AS similarity "
        "FROM items ORDER BY embedding <=> %s LIMIT %s",
        (vec, vec, limit),
    ).fetchall()
