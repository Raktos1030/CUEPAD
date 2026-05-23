"""System tray icon (Windows-focused, gracefully degrades elsewhere)."""
import threading
from pathlib import Path
from typing import Callable


class SystemTray:
    def __init__(
        self,
        on_show: Callable,
        on_quit: Callable,
        icon_path: Path | None = None,
        title: str = "Q-Pad",
    ):
        self._on_show = on_show
        self._on_quit = on_quit
        self._icon_path = icon_path
        self._title = title
        self._icon = None
        self._thread = None

    def start(self):
        try:
            import pystray
        except Exception:
            return False

        try:
            img = self._load_icon()

            def show_cb(icon, item):
                try:
                    self._on_show()
                except Exception:
                    pass

            def quit_cb(icon, item):
                try:
                    icon.stop()
                except Exception:
                    pass
                try:
                    self._on_quit()
                except Exception:
                    pass

            menu = pystray.Menu(
                pystray.MenuItem(f"Ouvrir {self._title}", show_cb, default=True),
                pystray.MenuItem("Quitter", quit_cb),
            )
            self._icon = pystray.Icon("cuepad", img, self._title, menu)
            self._thread = threading.Thread(target=self._icon.run, daemon=True)
            self._thread.start()
            return True
        except Exception:
            return False

    def stop(self):
        try:
            if self._icon:
                self._icon.stop()
        except Exception:
            pass

    def _load_icon(self):
        from PIL import Image
        if self._icon_path and Path(self._icon_path).exists():
            try:
                return Image.open(self._icon_path)
            except Exception:
                pass
        return self._default_icon()

    @staticmethod
    def _default_icon():
        from PIL import Image, ImageDraw
        size = 64
        img = Image.new("RGBA", (size, size), (8, 7, 13, 255))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((6, 6, 58, 58), radius=14, fill=(255, 59, 107, 255))
        d.rounded_rectangle((20, 20, 44, 44), radius=6, fill=(139, 92, 246, 255))
        return img
