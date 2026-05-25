"""认证模块 — 注册/登录/登出 + login_required 装饰器。

会话策略：滚动 30 天过期。每次请求刷新 last_active_at，
连续 30 天不活跃才需要重新登录。
"""

import functools
import re
from datetime import datetime, timedelta

import bcrypt
from flask import Blueprint, request, jsonify, session, g

from ai_trader.database import query, fetchone, execute

auth_bp = Blueprint("auth", __name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_SESSION_DAYS = 30


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_user_id() -> int:
    return session.get("user_id", 0)


def _is_logged_in() -> bool:
    return _get_user_id() > 0


# ---------------------------------------------------------------------------
# 装饰器
# ---------------------------------------------------------------------------

def login_required(f):
    """要求登录的装饰器。未登录返回 401，已登录刷新 last_active_at。"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        user_id = _get_user_id()
        if not user_id:
            return jsonify({"ok": False, "error": "请先登录", "code": "UNAUTH"}), 401
        # 滚动续期
        execute("UPDATE users SET last_active_at=%s WHERE id=%s", (_now(), user_id))
        session.permanent = True
        g.user_id = user_id
        return f(*args, **kwargs)
    return wrapper


def current_user() -> dict | None:
    """获取当前登录用户信息。"""
    uid = _get_user_id()
    if not uid:
        return None
    return fetchone("SELECT id, email, nickname, last_active_at, created_at FROM users WHERE id=%s", (uid,))


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(force=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()
    nickname = (body.get("nickname") or "").strip() or email.split("@")[0]

    if not email or not password:
        return jsonify({"ok": False, "error": "邮箱和密码不能为空"}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({"ok": False, "error": "邮箱格式不正确"}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "密码至少6位"}), 400

    existing = fetchone("SELECT id FROM users WHERE email=%s", (email,))
    if existing:
        return jsonify({"ok": False, "error": "该邮箱已注册"}), 409

    user_id = execute(
        "INSERT INTO users (email, password_hash, nickname, last_active_at) VALUES (%s, %s, %s, %s)",
        (email, _hash_password(password), nickname, _now()),
    )

    session.permanent = True
    session["user_id"] = user_id

    return jsonify({"ok": True, "user": {"id": user_id, "email": email, "nickname": nickname}})


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    body = request.get_json(force=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"ok": False, "error": "邮箱和密码不能为空"}), 400

    user = fetchone(
        "SELECT id, email, password_hash, nickname FROM users WHERE email=%s", (email,)
    )
    if not user or not _check_password(password, user["password_hash"]):
        return jsonify({"ok": False, "error": "邮箱或密码错误"}), 401

    session.permanent = True
    session["user_id"] = user["id"]
    execute("UPDATE users SET last_active_at=%s WHERE id=%s", (_now(), user["id"]))

    return jsonify({"ok": True, "user": {
        "id": user["id"], "email": user["email"], "nickname": user["nickname"],
    }})


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/auth/me")
def me():
    user = current_user()
    if not user:
        return jsonify({"ok": False, "error": "未登录"}), 401
    return jsonify({"ok": True, "user": {
        "id": user["id"], "email": user["email"], "nickname": user["nickname"],
        "created_at": str(user["created_at"]),
    }})
