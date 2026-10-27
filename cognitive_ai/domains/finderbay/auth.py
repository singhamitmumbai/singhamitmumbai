"""Multi-user authentication and authorization for Finderbay actions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from typing import Any, Optional

from ...core.records import utc_now


@dataclass
class User:
    user_id: str
    name: str
    roles: list[str] = field(default_factory=list)
    password_hash: str = ""
    salt: str = ""
    active: bool = True


PERMISSIONS = {
    "viewer": {"read"},
    "clerk": {"read", "draft", "task_write", "write_alias"},
    "accountant": {"read", "draft", "task_write", "payment_write", "invoice_write", "po_write"},
    "manager": {
        "read",
        "draft",
        "task_write",
        "payment_write",
        "invoice_write",
        "contract_write",
        "inventory_write",
        "write_alias",
        "po_write",
    },
    "admin": {
        "read",
        "draft",
        "task_write",
        "payment_write",
        "invoice_write",
        "contract_write",
        "inventory_write",
        "write_alias",
        "po_write",
        "send",
        "admin",
    },
}


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


class AuthService:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self._seed()

    def _seed(self) -> None:
        self.create_user("viewer1", "Viewer One", ["viewer"], "view-pass")
        self.create_user("clerk1", "Clerk One", ["clerk"], "clerk-pass")
        self.create_user("acct1", "Accountant One", ["accountant"], "acct-pass")
        self.create_user("mgr1", "Manager One", ["manager"], "mgr-pass")
        self.create_user("admin1", "Admin One", ["admin"], "admin-pass")

    def create_user(self, user_id: str, name: str, roles: list[str], password: str) -> User:
        salt = secrets.token_hex(8)
        user = User(
            user_id=user_id,
            name=name,
            roles=roles,
            salt=salt,
            password_hash=_hash_password(password, salt),
        )
        self.users[user_id] = user
        return user

    def login(self, user_id: str, password: str) -> Optional[str]:
        user = self.users.get(user_id)
        if not user or not user.active:
            return None
        if not hmac.compare_digest(user.password_hash, _hash_password(password, user.salt)):
            return None
        token = secrets.token_hex(16)
        self.sessions[token] = {
            "user_id": user_id,
            "roles": list(user.roles),
            "created_at": utc_now(),
        }
        return token

    def logout(self, token: str) -> bool:
        return self.sessions.pop(token, None) is not None

    def current_user(self, token: Optional[str]) -> Optional[User]:
        if not token or token not in self.sessions:
            return None
        uid = self.sessions[token]["user_id"]
        return self.users.get(uid)

    def permissions_for(self, user: Optional[User]) -> set[str]:
        if not user:
            return set()
        perms: set[str] = set()
        for role in user.roles:
            perms |= PERMISSIONS.get(role, set())
        return perms

    def authorize(self, token: Optional[str], permission: str) -> bool:
        user = self.current_user(token)
        return permission in self.permissions_for(user)

    def require(self, token: Optional[str], permission: str) -> User:
        user = self.current_user(token)
        if not user or permission not in self.permissions_for(user):
            raise PermissionError(f"permission_denied:{permission}")
        return user

    def list_users(self) -> list[dict[str, Any]]:
        """Multi-user directory (no password hashes)."""
        return [
            {
                "user_id": u.user_id,
                "name": u.name,
                "roles": list(u.roles),
                "active": u.active,
                "permissions": sorted(self.permissions_for(u)),
            }
            for u in self.users.values()
        ]

    def session_info(self, token: Optional[str]) -> Optional[dict[str, Any]]:
        if not token or token not in self.sessions:
            return None
        sess = dict(self.sessions[token])
        user = self.users.get(sess.get("user_id") or "")
        sess["permissions"] = sorted(self.permissions_for(user)) if user else []
        sess["name"] = user.name if user else None
        return sess

    def active_session_count(self) -> int:
        return len(self.sessions)
