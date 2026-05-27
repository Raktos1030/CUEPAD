"""Persistent JSON settings for Q-Pad."""
import json
import threading
from pathlib import Path


class Settings:
    DEFAULTS = {
        "output_main": None,        # device id (int) or None for default
        "output_monitor": None,
        "volume_main": 1.0,
        "volume_monitor": 0.7,
        "monitor_enabled": True,
        "monitor_muted": False,
        "hotkeys_enabled": True,
        "library_dir": None,        # None = use default downloads dir
        "active_tab": "soundboard",
        "sounds": {},               # filename -> {volume, hotkey, color}
    }

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return dict(self.DEFAULTS)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in self.DEFAULTS.items():
                data.setdefault(k, v)
            return data
        except Exception:
            return dict(self.DEFAULTS)

    def save(self):
        with self._lock:
            try:
                self.path.write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

    def update(self, patch: dict):
        self._data.update(patch)
        self.save()

    def sound(self, filename: str) -> dict:
        sounds = self._data.setdefault("sounds", {})
        return sounds.setdefault(filename, {})

    def set_sound(self, filename: str, patch: dict):
        s = self.sound(filename)
        s.update(patch)
        self.save()

    def delete_sound(self, filename: str):
        self._data.get("sounds", {}).pop(filename, None)
        self.save()

    def rename_sound(self, old: str, new: str):
        sounds = self._data.get("sounds", {})
        if old in sounds:
            sounds[new] = sounds.pop(old)
            self.save()

    def all(self) -> dict:
        return dict(self._data)

    # ─── Per-voice (AI voice changer) params ───────────────────────────────
    def voice_params(self, voice_name: str) -> dict:
        voices = self._data.setdefault("voice_params", {})
        return voices.get(voice_name, {})

    def set_voice_params(self, voice_name: str, patch: dict):
        voices = self._data.setdefault("voice_params", {})
        cur = voices.setdefault(voice_name, {})
        cur.update(patch)
        self.save()

    def delete_voice_params(self, voice_name: str):
        self._data.get("voice_params", {}).pop(voice_name, None)
        self.save()
