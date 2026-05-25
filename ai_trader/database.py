"""MySQL 数据库连接管理 — 基于 pymysql。

用法：
    from ai_trader.database import query, execute, fetchone, get_db

    rows = query("SELECT * FROM paper_trades WHERE user_id=%s", (user_id,))
    execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, h))
"""

import os
import pymysql
from contextlib import contextmanager


_DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "ichigo"),
    "database": os.environ.get("MYSQL_DATABASE", "alphamind_trader"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
}


@contextmanager
def get_db():
    """获取数据库连接，使用完毕后自动归还。"""
    conn = pymysql.connect(**_DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def query(sql: str, params=None) -> list[dict]:
    """执行 SELECT 查询，返回 dict 列表。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()


def fetchone(sql: str, params=None) -> dict | None:
    """执行 SELECT 查询，返回单条 dict 或 None。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def execute(sql: str, params=None) -> int:
    """执行 INSERT/UPDATE/DELETE，返回 lastrowid。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.lastrowid


def executemany(sql: str, params_list: list) -> int:
    """批量执行 INSERT/UPDATE/DELETE，返回影响行数。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            count = cur.executemany(sql, params_list)
            return count
