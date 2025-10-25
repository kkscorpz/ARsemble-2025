# storage.py
import json
import threading
from pathlib import Path

_lock = threading.RLock()


def _ensure_dir(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str, default=None):
    default = default or {}
    p = Path(path)
    with _lock:
        if not p.exists():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default


def save_json(path: str, data):
    p = Path(path)
    with _lock:
        _ensure_dir(p)
        p.write_text(json.dumps(data, indent=2,
                     ensure_ascii=False), encoding="utf-8")
