from contextlib import contextmanager

import psycopg2

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
