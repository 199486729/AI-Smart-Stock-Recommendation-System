"""认证模块 — 邮箱密码注册、登录、登出与访问控制。"""

import functools
import re
from datetime import datetime

import bcrypt
from flask import Blueprint, g, jsonify, request, session

from ai_trader.database import execute, fetchone

auth_bp = Blueprint("auth", __name__)

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (TypeError, ValueError, AttributeError):
        return False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_user_id() -> int:
    return session.get("user_id", 0)


def login_required(f):
    """要求用户登录，并在访问时刷新最后活跃时间。"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        user_id = _get_user_id()
        if not user_id:
            return jsonify({"ok": False, "error": "请先登录", "code": "UNAUTH"}), 401

        user = fetchone("SELECT id FROM users WHERE id=%s", (user_id,))
        if not user:
            session.pop("user_id", None)
            return jsonify({"ok": False, "error": "登录状态已失效", "code": "UNAUTH"}), 401

        execute("UPDATE users SET last_active_at=%s WHERE id=%s", (_now(), user_id))
        session.permanent = True
        g.user_id = user_id
        return f(*args, **kwargs)

    return wrapper


def current_user() -> dict | None:
    user_id = _get_user_id()
    if not user_id:
        return None
    return fetchone(
        "SELECT id, email, nickname, last_active_at, created_at FROM users WHERE id=%s",
        (user_id,),
    )


@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    body = request.get_json(force=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    nickname = (body.get("nickname") or "").strip() or email.split("@")[0]

    if not email or not password:
        return jsonify({"ok": False, "error": "邮箱和密码不能为空"}), 400
    if not _EMAIL_RE.match(email):
        return jsonify({"ok": False, "error": "邮箱格式不正确"}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "error": "密码至少6位"}), 400
    if len(nickname) > 100:
        return jsonify({"ok": False, "error": "昵称不能超过100个字符"}), 400
    if fetchone("SELECT id FROM users WHERE email=%s", (email,)):
        return jsonify({"ok": False, "error": "该邮箱已注册"}), 409

    try:
        user_id = execute(
            "INSERT INTO users (email, password_hash, nickname, last_active_at) "
            "VALUES (%s, %s, %s, %s)",
            (email, _hash_password(password), nickname, _now()),
        )
    except Exception as exc:
        if "Duplicate entry" in str(exc):
            return jsonify({"ok": False, "error": "该邮箱已注册"}), 409
        raise

    session.clear()
    session.permanent = True
    session["user_id"] = user_id
    return jsonify({"ok": True, "user": {"id": user_id, "email": email, "nickname": nickname}})


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    body = request.get_json(force=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "邮箱和密码不能为空"}), 400

    user = fetchone(
        "SELECT id, email, password_hash, nickname FROM users WHERE email=%s",
        (email,),
    )
    if not user or not _check_password(password, user.get("password_hash")):
        return jsonify({"ok": False, "error": "邮箱或密码错误"}), 401

    session.clear()
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
        "id": user["id"],
        "email": user["email"],
        "nickname": user["nickname"],
        "created_at": str(user["created_at"]),
    }})
