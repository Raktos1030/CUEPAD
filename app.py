"""Q-Pad — Flask routes layer.

The actual logic lives in:
- converter.py  : YT → audio downloads + history
- library.py    : audio library scanning
- audio_engine.py : multi-output playback
- hotkeys.py    : global hotkey manager
- settings.py   : persistent JSON settings

main.py wires it all together and calls `configure()` here with the services.
"""
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

if getattr(sys, "frozen", False):
    _template_dir = Path(sys._MEIPASS) / "templates"
else:
    _template_dir = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(_template_dir))

# Services bag — wired in main.py via configure()
services: dict = {}


def configure(*, converter, library, audio, hotkeys, settings, on_show, on_quit):
    services["converter"] = converter
    services["library"] = library
    services["audio"] = audio
    services["hotkeys"] = hotkeys
    services["settings"] = settings
    services["on_show"] = on_show
    services["on_quit"] = on_quit


# ─────────────────────────── Pages ────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────── Convert ──────────────────────────────────────

@app.route("/info", methods=["POST"])
def info():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400
    try:
        return jsonify(services["converter"].fetch_info(url))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400
    job_id = services["converter"].submit(
        url=url,
        start=(data.get("start") or "").strip(),
        end=(data.get("end") or "").strip(),
        name=(data.get("name") or "").strip(),
        fmt=(data.get("format") or "mp3"),
        quality=(data.get("quality") or "320"),
        original=bool(data.get("original")),
    )
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    s = services["converter"].job_status(job_id)
    if not s:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(s)


@app.route("/download/<job_id>")
def download(job_id: str):
    f = services["converter"].job_file(job_id)
    if not f:
        return "Fichier non prêt", 404
    return send_file(str(f), as_attachment=True, download_name=f.name)


# ─────────────────────────── History ──────────────────────────────────────

@app.route("/history")
def history():
    conv = services["converter"]
    return jsonify({
        "items": conv.read_history(),
        "downloads_dir": str(conv.downloads),
    })


@app.route("/history/clear", methods=["POST"])
def history_clear():
    services["converter"].clear_history()
    return jsonify({"ok": True})


@app.route("/history/file/<path:filename>")
def history_file(filename: str):
    target = services["library"].get_path(filename)
    if not target:
        return "Fichier introuvable", 404
    return send_file(str(target), as_attachment=True, download_name=target.name)


# ─────────────────────────── Library / sounds ─────────────────────────────

@app.route("/sounds")
def sounds_list():
    lib = services["library"]
    sett = services["settings"]
    items = lib.list()
    cfg = sett.get("sounds", {})
    for it in items:
        it["config"] = cfg.get(it["filename"], {})
    return jsonify({"items": items, "library_dir": str(lib.root)})


@app.route("/sounds/play", methods=["POST"])
def sound_play():
    data = request.get_json(silent=True) or {}
    filename = (data.get("filename") or "").strip()
    lib = services["library"]
    path = lib.get_path(filename)
    if not path:
        return jsonify({"error": "Fichier introuvable"}), 404

    sett = services["settings"]
    per_vol = float(sett.sound(filename).get("volume", 1.0))

    ok, err, pb_id = services["audio"].play(
        str(path),
        device_main=sett.get("output_main"),
        device_monitor=sett.get("output_monitor"),
        per_sound_gain=per_vol,
        monitor_enabled=bool(sett.get("monitor_enabled", True)),
        name=Path(filename).stem,
    )
    return jsonify({"ok": ok, "error": err, "id": pb_id})


@app.route("/sounds/stop", methods=["POST"])
def sound_stop():
    services["audio"].stop_all()
    return jsonify({"ok": True})


# ─────────────────────────── Voices (live playback control) ───────────────

@app.route("/voices")
def voices_list():
    return jsonify({"items": services["audio"].list_playbacks()})


@app.route("/voices/<pb_id>/pause", methods=["POST"])
def voice_pause(pb_id):
    return jsonify({"ok": services["audio"].pause(pb_id)})


@app.route("/voices/<pb_id>/resume", methods=["POST"])
def voice_resume(pb_id):
    return jsonify({"ok": services["audio"].resume(pb_id)})


@app.route("/voices/<pb_id>/stop", methods=["POST"])
def voice_stop(pb_id):
    return jsonify({"ok": services["audio"].stop(pb_id)})


@app.route("/voices/<pb_id>/seek", methods=["POST"])
def voice_seek(pb_id):
    data = request.get_json(silent=True) or {}
    try:
        pos = float(data.get("pos", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid pos"}), 400
    return jsonify({"ok": services["audio"].seek(pb_id, pos)})


@app.route("/voices/<pb_id>/speed", methods=["POST"])
def voice_speed(pb_id):
    data = request.get_json(silent=True) or {}
    try:
        speed = float(data.get("speed", 1.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid speed"}), 400
    ok = services["audio"].set_speed(pb_id, speed)
    return jsonify({"ok": ok})


@app.route("/voices/pause-all", methods=["POST"])
def voices_pause_all():
    services["audio"].pause_all()
    return jsonify({"ok": True})


@app.route("/voices/resume-all", methods=["POST"])
def voices_resume_all():
    services["audio"].resume_all()
    return jsonify({"ok": True})


@app.route("/sounds/open-folder", methods=["POST"])
def sounds_open_folder():
    import os
    import subprocess
    path = str(services["library"].root)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/sounds/<path:filename>", methods=["DELETE"])
def sound_delete(filename: str):
    ok = services["library"].delete(filename)
    if ok:
        services["settings"].delete_sound(filename)
        _rebind_hotkeys()
    return jsonify({"ok": ok})


@app.route("/sounds/<path:filename>/rename", methods=["POST"])
def sound_rename(filename: str):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("new") or "").strip()
    if not new_name:
        return jsonify({"error": "Nom manquant"}), 400
    result = services["library"].rename(filename, new_name)
    if not result:
        return jsonify({"error": "Renommage impossible"}), 400
    services["settings"].rename_sound(filename, result)
    _rebind_hotkeys()
    return jsonify({"ok": True, "filename": result})


@app.route("/sounds/<path:filename>/config", methods=["POST"])
def sound_config(filename: str):
    data = request.get_json(silent=True) or {}
    patch = {}
    if "volume" in data:
        try:
            patch["volume"] = max(0.0, min(2.0, float(data["volume"])))
        except (TypeError, ValueError):
            pass
    if "hotkey" in data:
        patch["hotkey"] = (data["hotkey"] or "").strip()
    if "color" in data:
        patch["color"] = (data["color"] or "").strip()
    services["settings"].set_sound(filename, patch)
    if "hotkey" in patch:
        _rebind_hotkeys()
    return jsonify({"ok": True})


@app.route("/sounds/<path:filename>/file")
def sound_file(filename: str):
    p = services["library"].get_path(filename)
    if not p:
        return "Fichier introuvable", 404
    return send_file(str(p), as_attachment=False, download_name=p.name)


# ─────────────────────────── Devices ──────────────────────────────────────

@app.route("/devices")
def devices():
    return jsonify(services["audio"].list_devices())


# ─────────────────────────── Settings ─────────────────────────────────────

@app.route("/settings")
def settings_get():
    return jsonify(services["settings"].all())


@app.route("/settings", methods=["POST"])
def settings_set():
    data = request.get_json(silent=True) or {}
    sett = services["settings"]
    allowed = {
        "output_main", "output_monitor",
        "volume_main", "volume_monitor",
        "monitor_enabled", "monitor_muted",
        "hotkeys_enabled", "library_dir", "active_tab",
    }
    patch = {k: v for k, v in data.items() if k in allowed}
    if "hotkeys_enabled" in patch:
        services["hotkeys"].set_enabled(bool(patch["hotkeys_enabled"]))
    # Live-propagate audio state so currently playing voices react immediately.
    audio = services["audio"]
    if "volume_main" in patch:
        audio.set_global_main(patch["volume_main"])
    if "volume_monitor" in patch:
        audio.set_global_monitor(patch["volume_monitor"])
    if "monitor_muted" in patch:
        audio.set_monitor_muted(patch["monitor_muted"])
    # The UI toggle "Monitor activé" only sets monitor_enabled, but users
    # expect it to silence the monitor live as well — not just gate the next
    # play() call.
    if "monitor_enabled" in patch:
        audio.set_monitor_muted(not patch["monitor_enabled"])
    sett.update(patch)
    return jsonify({"ok": True, "settings": sett.all()})


# ─────────────────────────── App control ──────────────────────────────────

@app.route("/quit", methods=["POST"])
def quit_app():
    cb = services.get("on_quit")
    if cb:
        try:
            cb()
        except Exception:
            pass
    return jsonify({"ok": True})


# ─────────────────────────── Helpers ──────────────────────────────────────

def _rebind_hotkeys():
    """Re-register all hotkeys based on the current settings + library."""
    sett = services["settings"]
    lib = services["library"]
    hk = services["hotkeys"]
    if not hk.available():
        return

    known = {it["filename"] for it in lib.list()}
    mapping = {}
    for filename, cfg in sett.get("sounds", {}).items():
        if filename not in known:
            continue
        combo = (cfg or {}).get("hotkey")
        if not combo:
            continue
        mapping[combo] = _play_callback(filename)
    hk.rebind_all(mapping)


def _play_callback(filename: str):
    def cb():
        lib = services["library"]
        sett = services["settings"]
        audio = services["audio"]
        path = lib.get_path(filename)
        if not path:
            return
        per_vol = float(sett.sound(filename).get("volume", 1.0))
        audio.play(
            str(path),
            device_main=sett.get("output_main"),
            device_monitor=sett.get("output_monitor"),
            per_sound_gain=per_vol,
            monitor_enabled=bool(sett.get("monitor_enabled", True)),
            name=Path(filename).stem,
        )  # tuple result ignored — hotkeys are fire-and-forget
    return cb


def rebind_hotkeys():
    """Public entry point — called from main.py on startup and on library change."""
    _rebind_hotkeys()
