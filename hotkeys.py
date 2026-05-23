"""Global hotkey manager backed by pynput."""
import threading
from typing import Callable


def normalize_combo(combo: str) -> str:
    """Convert user-friendly 'Ctrl+Shift+1' to pynput '<ctrl>+<shift>+1'."""
    if not combo:
        return ""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    out = []
    mods = {"ctrl", "control", "alt", "shift", "cmd", "win", "super"}
    for p in parts:
        if p in mods:
            p = "ctrl" if p == "control" else p
            p = "cmd" if p in ("win", "super") else p
            out.append(f"<{p}>")
        elif len(p) == 1:
            out.append(p)
        elif p.startswith("f") and p[1:].isdigit():
            out.append(f"<{p}>")
        else:
            out.append(f"<{p}>")
    return "+".join(out)


def _shift_letter_variants(combo: str) -> list[str]:
    """For '<shift>+a' yield ['<shift>+A']. Empty otherwise.

    Works around pynput delivering the uppercase KeyCode when Shift is held,
    which never matches a hotkey registered with the lowercase letter.
    """
    parts = combo.split("+")
    if "<shift>" not in parts:
        return []
    out = []
    for i, p in enumerate(parts):
        if len(p) == 1 and p.isalpha() and p == p.lower():
            variant = parts[:i] + [p.upper()] + parts[i + 1:]
            out.append("+".join(variant))
    return out


class HotkeyManager:
    def __init__(self):
        self._listener = None
        self._bindings: dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._enabled = True
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import pynput  # noqa: F401
            return True
        except Exception:
            return False

    def available(self) -> bool:
        return self._available

    def set_enabled(self, enabled: bool):
        self._enabled = bool(enabled)
        self._refresh()

    def bind(self, combo: str, callback: Callable):
        norm = normalize_combo(combo)
        if not norm:
            return
        with self._lock:
            self._bindings[norm] = callback
        self._refresh()

    def unbind(self, combo: str):
        norm = normalize_combo(combo)
        with self._lock:
            self._bindings.pop(norm, None)
        self._refresh()

    def clear(self):
        with self._lock:
            self._bindings.clear()
        self._refresh()

    def rebind_all(self, mapping: dict[str, Callable]):
        with self._lock:
            self._bindings = {
                normalize_combo(k): v for k, v in mapping.items() if k
            }
        self._refresh()

    def _refresh(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

        if not self._enabled or not self._bindings or not self._available:
            return

        try:
            from pynput import keyboard
            mapping = {}
            for combo, cb in self._bindings.items():
                wrapped = self._wrap(cb)
                mapping[combo] = wrapped
                # pynput quirk: when Shift is held, the OS reports the
                # uppercase char, so '<shift>+a' never matches. Register
                # both the lower and upper variants for shift+letter.
                for variant in _shift_letter_variants(combo):
                    mapping[variant] = wrapped
            self._listener = keyboard.GlobalHotKeys(mapping)
            self._listener.start()
        except Exception:
            self._listener = None

    def _wrap(self, cb: Callable):
        def w():
            if self._enabled:
                try:
                    cb()
                except Exception:
                    pass
        return w

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
