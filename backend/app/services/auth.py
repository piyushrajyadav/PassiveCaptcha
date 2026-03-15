from __future__ import annotations

import hashlib
import secrets
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "app.db"
DEFAULT_ADMIN_EMAIL = "admin@admin.com"
DEFAULT_ADMIN_PASSWORD = "password"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_auth_tables() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL
            )
            """
        )
        connection.commit()


def seed_default_admin() -> None:
    ensure_auth_tables()
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT id FROM users WHERE email = ?",
            (DEFAULT_ADMIN_EMAIL,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO users (email, password_hash, display_name)
                VALUES (?, ?, ?)
                """,
                (
                    DEFAULT_ADMIN_EMAIL,
                    hash_password(DEFAULT_ADMIN_PASSWORD),
                    "Admin Operator",
                ),
            )
            connection.commit()


def authenticate_user(email: str, password: str) -> dict[str, str] | None:
    ensure_auth_tables()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT email, display_name, password_hash
            FROM users
            WHERE email = ?
            """,
            (email.lower().strip(),),
        ).fetchone()

    if row is None:
        return None

    if not secrets.compare_digest(row["password_hash"], hash_password(password)):
        return None

    return {
        "email": row["email"],
        "display_name": row["display_name"],
    }
