from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from backend.config import settings


@contextmanager
def get_connection():
    conn = psycopg2.connect(settings.database_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with open("database/init.sql", encoding="utf-8") as f:
        sql = f.read()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def cache_count() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM cache_entries")
            return cur.fetchone()[0]


def insert_cache_entry(question: str, answer: str, embedding: list[float]) -> None:
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cache_entries (question, answer, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (question, answer, embedding_str),
            )


def find_similar_cache(embedding: list[float], threshold: float) -> dict | None:
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT question, answer,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM cache_entries
                ORDER BY embedding <=> %s::vector
                LIMIT 1
                """,
                (embedding_str, embedding_str),
            )
            row = cur.fetchone()
            if row and row["similarity"] >= threshold:
                return dict(row)
    return None
