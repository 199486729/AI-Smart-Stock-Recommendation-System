"""数据库初始化脚本 — 创建 AlphaMind 注册登录版所需的表。

用法：
    python -m ai_trader.init_db          # 创建所有表
    python -m ai_trader.init_db --drop   # 先删再建（危险！）
"""

import argparse
import os
import sys

# 确保 ai_trader 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trader.database import get_db, fetchone

SCHEMA = """
-- ============================================================================
-- 用户表（邮箱密码注册）
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    nickname VARCHAR(100) NOT NULL DEFAULT '',
    last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_last_active (last_active_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 聊天会话
-- ============================================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(200) DEFAULT '新对话',
    summary TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 聊天消息
-- ============================================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    role VARCHAR(16) NOT NULL COMMENT 'user|assistant|system',
    content TEXT NOT NULL,
    tool_calls_json TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 推荐快照
-- ============================================================================
CREATE TABLE IF NOT EXISTS recommendation_snapshots (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    market_state_json LONGTEXT NOT NULL,
    sectors_json LONGTEXT NOT NULL,
    recommendations_json LONGTEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 用户关键事实（长期记忆）
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_key_facts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    fact TEXT NOT NULL,
    source_session_id INT DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 模拟交易记录
-- ============================================================================
CREATE TABLE IF NOT EXISTS paper_trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    code VARCHAR(16) NOT NULL,
    name VARCHAR(64) DEFAULT '',
    side VARCHAR(8) NOT NULL COMMENT 'buy|sell',
    price DECIMAL(12, 4) NOT NULL,
    quantity INT NOT NULL DEFAULT 0,
    amount DECIMAL(16, 2) NOT NULL,
    notes TEXT DEFAULT NULL,
    INDEX idx_user (user_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 真实交易记录
-- ============================================================================
CREATE TABLE IF NOT EXISTS real_trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    code VARCHAR(16) NOT NULL,
    name VARCHAR(64) DEFAULT '',
    side VARCHAR(8) NOT NULL COMMENT 'buy|sell',
    price DECIMAL(12, 4) NOT NULL,
    quantity INT NOT NULL DEFAULT 0,
    amount DECIMAL(16, 2) NOT NULL,
    notes TEXT DEFAULT NULL,
    INDEX idx_user (user_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

DROP_SQL = """
DROP TABLE IF EXISTS real_trades;
DROP TABLE IF EXISTS paper_trades;
DROP TABLE IF EXISTS user_key_facts;
DROP TABLE IF EXISTS recommendation_snapshots;
DROP TABLE IF EXISTS chat_messages;
DROP TABLE IF EXISTS chat_sessions;
DROP TABLE IF EXISTS users;
"""


def init_db(drop_first: bool = False):
    """创建所有表。"""
    with get_db() as conn:
        with conn.cursor() as cur:
            if drop_first:
                print("⚠ 删除所有表...")
                for stmt in DROP_SQL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
                print("  已删除。")

            for stmt in SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)

            # 兼容曾经使用邀请码表结构的数据库：保留旧用户记录，只补充注册字段。
            cur.execute("SHOW COLUMNS FROM users")
            user_columns = {row["Field"] for row in cur.fetchall()}
            if "email" not in user_columns:
                cur.execute("ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL")
            if "password_hash" not in user_columns:
                cur.execute("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) DEFAULT NULL")

            # 邀请码旧表的字段长度更短；保留旧用户的 NULL 登录字段并扩展容量。
            cur.execute("ALTER TABLE users MODIFY COLUMN email VARCHAR(255) DEFAULT NULL")
            cur.execute("ALTER TABLE users MODIFY COLUMN password_hash VARCHAR(255) DEFAULT NULL")
            cur.execute("ALTER TABLE users MODIFY COLUMN nickname VARCHAR(100) DEFAULT ''")

            cur.execute("SHOW INDEX FROM users WHERE Key_name='email'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE users ADD UNIQUE INDEX email (email)")

    print("✓ 数据库初始化完成。")

    # 验证
    tables = ["users", "chat_sessions", "chat_messages",
              "recommendation_snapshots", "user_key_facts", "paper_trades", "real_trades"]
    for t in tables:
        row = fetchone(f"SELECT COUNT(*) AS c FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s", (t,))
        status = "✓" if row and row["c"] > 0 else "✗"
        print(f"  {status} {t}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初始化 AlphaMind 数据库")
    parser.add_argument("--drop", action="store_true", help="先删除所有表再重新创建（危险！）")
    args = parser.parse_args()

    if args.drop:
        confirm = input("确认删除所有数据并重建？输入 yes 继续: ")
        if confirm.lower() != "yes":
            print("已取消。")
            sys.exit(0)

    init_db(drop_first=args.drop)
