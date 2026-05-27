"""Q-Pad — desktop entry point.

Boots Flask, PyWebView, the global hotkey listener, and the system tray.
"""
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path


# ─── Path resolution (PyInstaller vs dev) ───────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
    EXE_DIR = BASE_DIR


def _find_ffmpeg() -> str:
    """Locate ffmpeg, preferring bundled binaries over PATH.

    Returns an absolute path if found, otherwise the bare command name so
    subprocess will at least try PATH at call time.
    """
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidates = [
        BASE_DIR / exe,                    # PyInstaller bundle root / contents dir
        EXE_DIR / exe,                     # Next to the launched executable
        EXE_DIR / "_internal" / exe,       # PyInstaller 6+ onedir layout
        BASE_DIR / "_internal" / exe,
        BASE_DIR / "ffmpeg" / exe,         # Dev checkout: ./ffmpeg/ffmpeg.exe
        EXE_DIR / "ffmpeg" / exe,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Last resort: scan the install / project tree. Cheap one-time cost at
    # startup, and it catches the layouts our hand-picked candidates miss
    # (different PyInstaller versions move the contents dir around).
    for root in (EXE_DIR, BASE_DIR):
        try:
            for c in root.rglob(exe):
                if c.is_file():
                    return str(c)
        except Exception:
            pass

    found = shutil.which("ffmpeg")
    return found or exe


FFMPEG_CMD = _find_ffmpeg()

DOWNLOADS_DIR = Path.home() / "Downloads" / "Q-Pad Soundboard"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

APP_DATA = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "Q-Pad"
APP_DATA.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = APP_DATA / "settings.json"

ICON_PATH = BASE_DIR / "icon.ico" if (BASE_DIR / "icon.ico").exists() else None


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main():
    # ─── Settings + library directory ───────────────────────────────────────
    from settings import Settings
    settings = Settings(SETTINGS_FILE)
    library_dir = Path(settings.get("library_dir") or DOWNLOADS_DIR)
    library_dir.mkdir(parents=True, exist_ok=True)

    # ─── Services ───────────────────────────────────────────────────────────
    from converter import Converter
    from library import Library
    from audio_engine import AudioEngine
    from hotkeys import HotkeyManager
    from live_engine import LiveMicEngine
    from voice_changer import VoiceChanger
    from live_rvc import LiveRvcEngine

    converter = Converter(downloads_dir=library_dir, ffmpeg_cmd=FFMPEG_CMD)
    library = Library(root=library_dir)
    audio = AudioEngine(ffmpeg_path=FFMPEG_CMD)
    audio.set_global_main(settings.get("volume_main", 1.0))
    audio.set_global_monitor(settings.get("volume_monitor", 0.7))
    audio.set_monitor_muted(
        bool(settings.get("monitor_muted", False))
        or not bool(settings.get("monitor_enabled", True))
    )
    live = LiveMicEngine()
    voices_dir = APP_DATA / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    voice_changer = VoiceChanger(
        voices_dir=voices_dir,
        device_pref=settings.get("voice_ai_device_pref", "auto"),
    )
    live_rvc = LiveRvcEngine(voice_changer=voice_changer)
    hotkeys = HotkeyManager()
    hotkeys.set_enabled(bool(settings.get("hotkeys_enabled", True)))

    # ─── Flask wiring ───────────────────────────────────────────────────────
    import app as flask_app

    # window_ref will be set after webview.create_window
    window_ref: dict = {"win": None, "tray": None}

    def on_show():
        win = window_ref.get("win")
        if win is not None:
            try:
                win.show()
            except Exception:
                pass

    def on_quit():
        # Stop everything cleanly
        try:
            hotkeys.stop()
        except Exception:
            pass
        try:
            library.stop()
        except Exception:
            pass
        try:
            audio.stop_all()
        except Exception:
            pass
        tray = window_ref.get("tray")
        if tray is not None:
            try:
                tray.stop()
            except Exception:
                pass
        try:
            win = window_ref.get("win")
            if win is not None:
                win.destroy()
        except Exception:
            pass
        os._exit(0)

    flask_app.configure(
        converter=converter,
        library=library,
        audio=audio,
        live=live,
        voice_changer=voice_changer,
        live_rvc=live_rvc,
        hotkeys=hotkeys,
        settings=settings,
        on_show=on_show,
        on_quit=on_quit,
    )

    # ─── Library watcher: rebind hotkeys + invalidate frontend cache ────────
    def on_library_change():
        try:
            flask_app.rebind_hotkeys()
        except Exception:
            pass

    library.on_change = on_library_change
    converter.on_library_change = on_library_change
    library.start_watching()

    # Initial hotkey bind
    flask_app.rebind_hotkeys()

    # ─── Flask thread ───────────────────────────────────────────────────────
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    def run_flask():
        flask_app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    if not _wait_for_port(port):
        print("Flask failed to start", file=sys.stderr)
        sys.exit(1)

    # ─── System tray ────────────────────────────────────────────────────────
    from tray import SystemTray
    tray = SystemTray(
        on_show=on_show,
        on_quit=on_quit,
        icon_path=ICON_PATH,
        title="Q-Pad",
    )
    tray.start()
    window_ref["tray"] = tray

    # ─── Webview ────────────────────────────────────────────────────────────
    import webview

    class JsApi:
        def hide(self):
            win = window_ref.get("win")
            if win is not None:
                try:
                    win.hide()
                except Exception:
                    pass

    window = webview.create_window(
        "Q-Pad",
        url,
        width=1080,
        height=760,
        min_size=(820, 600),
        resizable=True,
        js_api=JsApi(),
    )
    window_ref["win"] = window

    def on_closing():
        # Minimize to tray instead of quitting
        if window_ref.get("tray") is not None:
            try:
                window.hide()
            except Exception:
                pass
            return False  # cancel the close
        return True

    window.events.closing += on_closing

    webview.start()


if __name__ == "__main__":
    main()
