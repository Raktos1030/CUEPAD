"""
Entry point for the desktop app (PyWebView + PyInstaller).
When run normally (python main.py), also works as a standalone launcher.
"""
import sys
import os
import threading
import time
import socket
from pathlib import Path

# --- Resolve paths for frozen (PyInstaller) vs dev mode ---
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys._MEIPASS)
    FFMPEG_CMD = str(BASE_DIR / "ffmpeg.exe")
else:
    BASE_DIR = Path(__file__).parent
    FFMPEG_CMD = "ffmpeg"

DOWNLOADS_DIR = Path.home() / "Downloads" / "YT-MP3"

# --- Configure app before importing Flask routes ---
import app as flask_app
flask_app.configure(ffmpeg_path=FFMPEG_CMD, downloads_dir=DOWNLOADS_DIR)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_flask(port: int):
    flask_app.app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def main():
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    flask_thread = threading.Thread(target=_run_flask, args=(port,), daemon=True)
    flask_thread.start()

    # Wait for Flask to be ready
    for _ in range(20):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.15)

    import webview
    window = webview.create_window(
        "YT → MP3",
        url,
        width=580,
        height=540,
        resizable=False,
        min_size=(580, 540),
    )
    webview.start()


if __name__ == "__main__":
    main()
