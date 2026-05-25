"""数据持久化 — 聊天记录、推荐快照、用户关键事实。"""

import json
from datetime import datetime

from ai_trader.database import query, fetchone, execute


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 聊天会话
# ---------------------------------------------------------------------------

def create_chat_session(user_id: int, title: str = "新对话") -> int:
    """新建会话，返回 session_id。"""
    return execute(
        "INSERT INTO chat_sessions (user_id, title) VALUES (%s, %s)",
        (user_id, title),
    )


def get_chat_sessions(user_id: int, limit: int = 20) -> list[dict]:
    """获取用户的会话列表。"""
    return query(
        "SELECT id, title, summary, created_at, updated_at "
        "FROM chat_sessions WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s",
        (user_id, limit),
    )


def update_session_title(session_id: int, title: str):
    """更新会话标题。"""
    execute("UPDATE chat_sessions SET title=%s WHERE id=%s", (title, session_id))


def update_session_summary(session_id: int, summary: str):
    """更新会话摘要。"""
    execute("UPDATE chat_sessions SET summary=%s WHERE id=%s", (summary, session_id))


def touch_session(session_id: int):
    """更新会话的 updated_at。"""
    execute("UPDATE chat_sessions SET updated_at=%s WHERE id=%s", (_now(), session_id))


# ---------------------------------------------------------------------------
# 聊天消息
# ---------------------------------------------------------------------------

def save_message(session_id: int, role: str, content: str,
                 tool_calls: list[dict] | None = None) -> int:
    """保存一条消息。role: user|assistant|system"""
    tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    msg_id = execute(
        "INSERT INTO chat_messages (session_id, role, content, tool_calls_json) "
        "VALUES (%s, %s, %s, %s)",
        (session_id, role, content, tc_json),
    )
    touch_session(session_id)
    return msg_id


def get_messages(session_id: int, limit: int = 100) -> list[dict]:
    """获取会话的消息列表。"""
    return query(
        "SELECT id, role, content, tool_calls_json, created_at "
        "FROM chat_messages WHERE session_id=%s ORDER BY id ASC LIMIT %s",
        (session_id, limit),
    )


# ---------------------------------------------------------------------------
# 推荐快照
# ---------------------------------------------------------------------------

def save_recommendation_snapshot(user_id: int, market_state: dict,
                                  sectors: list[dict],
                                  recommendations: list[dict]) -> int:
    """保存一次市场分析的完整快照。"""
    return execute(
        "INSERT INTO recommendation_snapshots "
        "(user_id, market_state_json, sectors_json, recommendations_json) "
        "VALUES (%s, %s, %s, %s)",
        (
            user_id,
            json.dumps(market_state, ensure_ascii=False, default=str),
            json.dumps(sectors, ensure_ascii=False, default=str),
            json.dumps(recommendations, ensure_ascii=False, default=str),
        ),
    )


def get_recommendation_snapshots(user_id: int, limit: int = 10) -> list[dict]:
    """获取用户最近的分析快照列表（只含元数据，不含完整JSON）。"""
    return query(
        "SELECT id, created_at, "
        "JSON_EXTRACT(market_state_json, '$.composite') AS market_composite, "
        "JSON_LENGTH(recommendations_json) AS rec_count "
        "FROM recommendation_snapshots "
        "WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
        (user_id, limit),
    )


def get_snapshot_detail(snapshot_id: int, user_id: int) -> dict | None:
    """获取单条快照完整数据。"""
    return fetchone(
        "SELECT * FROM recommendation_snapshots WHERE id=%s AND user_id=%s",
        (snapshot_id, user_id),
    )


# ---------------------------------------------------------------------------
# 用户关键事实
# ---------------------------------------------------------------------------

def save_fact(user_id: int, fact: str, source_session_id: int | None = None) -> int:
    """保存一条关键事实。去重：相同 fact 不重复插入。"""
    existing = fetchone(
        "SELECT id FROM user_key_facts WHERE user_id=%s AND fact=%s",
        (user_id, fact),
    )
    if existing:
        return existing["id"]
    return execute(
        "INSERT INTO user_key_facts (user_id, fact, source_session_id) VALUES (%s, %s, %s)",
        (user_id, fact, source_session_id),
    )


def get_facts(user_id: int) -> list[dict]:
    """获取用户所有关键事实。"""
    return query(
        "SELECT id, fact, source_session_id, created_at "
        "FROM user_key_facts WHERE user_id=%s ORDER BY id ASC",
        (user_id,),
    )
