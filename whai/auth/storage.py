"""Persistent auth profile storage for OAuth and rotating credentials."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from whai.configuration.user_config import get_config_dir

AUTH_PROFILES_FILENAME = "auth-profiles.json"


def get_auth_profiles_path() -> Path:
    """Return the auth profile store path."""
    return (get_config_dir() / AUTH_PROFILES_FILENAME).resolve()


@contextmanager
def _lock_file(lock_path: Path, timeout_seconds: float = 5.0) -> Iterator[None]:
    """Acquire a simple exclusive file lock using lockfile creation."""
    start = time.time()
    fd: Optional[int] = None
    while time.time() - start < timeout_seconds:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            time.sleep(0.05)

    if fd is None:
        raise TimeoutError(f"Timed out acquiring auth lock: {lock_path}")

    try:
        yield
    finally:
        try:
            os.close(fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()


def load_auth_profiles() -> Dict[str, Any]:
    """Load auth profiles from disk; return empty structure when missing."""
    path = get_auth_profiles_path()
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_auth_profiles(data: Dict[str, Any]) -> None:
    """Atomically save auth profiles to disk with a lightweight lock."""
    path = get_auth_profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with _lock_file(lock_path):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)


def get_openai_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """Get an OpenAI OAuth profile by id."""
    data = load_auth_profiles()
    openai = data.get("openai") or {}
    profiles = openai.get("profiles") or {}
    profile = profiles.get(profile_id)
    return profile if isinstance(profile, dict) else None


def upsert_openai_profile(profile_id: str, profile_data: Dict[str, Any]) -> None:
    """Insert or update an OpenAI OAuth profile."""
    data = load_auth_profiles()
    openai = data.setdefault("openai", {})
    profiles = openai.setdefault("profiles", {})
    profiles[profile_id] = profile_data
    save_auth_profiles(data)
