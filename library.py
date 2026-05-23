"""Audio library: scans and watches the sounds folder."""
import threading
from pathlib import Path
from typing import Callable

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".opus", ".ogg", ".flac", ".aac"}


class Library:
    def __init__(self, root: Path, on_change: Callable | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.on_change = on_change
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._snapshot: set[str] = set()

    def list(self) -> list[dict]:
        items = []
        try:
            entries = [p for p in self.root.iterdir() if p.is_file()]
        except Exception:
            return items
        entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in entries:
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            try:
                stat = p.stat()
                items.append({
                    "filename": p.name,
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                    "ext": p.suffix[1:].lower(),
                    "stem": p.stem,
                })
            except Exception:
                continue
        return items

    def get_path(self, filename: str) -> Path | None:
        safe = Path(filename).name
        p = self.root / safe
        return p if p.exists() else None

    def delete(self, filename: str) -> bool:
        p = self.get_path(filename)
        if not p:
            return False
        try:
            p.unlink()
            return True
        except Exception:
            return False

    def rename(self, old: str, new: str) -> str | None:
        src = self.get_path(old)
        if not src:
            return None
        ext = src.suffix
        safe = Path(new).name
        if not safe:
            return None
        if not safe.lower().endswith(ext.lower()):
            safe += ext
        dest = self.root / safe
        n = 1
        while dest.exists() and dest != src:
            dest = self.root / f"{Path(safe).stem} ({n}){ext}"
            n += 1
        try:
            src.rename(dest)
            return dest.name
        except Exception:
            return None

    def start_watching(self):
        if self._poll_thread:
            return
        self._snapshot = self._current_set()
        self._poll_thread = threading.Thread(target=self._poll, daemon=True)
        self._poll_thread.start()

    def _current_set(self) -> set[str]:
        try:
            return {
                p.name for p in self.root.iterdir()
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            }
        except Exception:
            return set()

    def _poll(self):
        while not self._stop.is_set():
            try:
                current = self._current_set()
                if current != self._snapshot:
                    self._snapshot = current
                    if self.on_change:
                        try:
                            self.on_change()
                        except Exception:
                            pass
            except Exception:
                pass
            self._stop.wait(2.0)

    def stop(self):
        self._stop.set()
