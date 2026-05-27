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


def configure(*, converter, library, audio, live, voice_changer, live_rvc, hotkeys, settings, on_show, on_quit):
    services["converter"] = converter
    services["library"] = library
    services["audio"] = audio
    services["live"] = live
    services["voice_changer"] = voice_changer
    services["live_rvc"] = live_rvc
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
    cfg = sett.sound(filename)
    per_vol = float(cfg.get("volume", 1.0))
    effects = cfg.get("effects") or None

    ok, err, pb_id = services["audio"].play(
        str(path),
        device_main=sett.get("output_main"),
        device_monitor=sett.get("output_monitor"),
        per_sound_gain=per_vol,
        monitor_enabled=bool(sett.get("monitor_enabled", True)),
        name=Path(filename).stem,
        effects=effects,
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


# ─────────────────────────── Effects (live + file) ─────────────────────────

@app.route("/effects/live/status")
def live_status():
    return jsonify(services["live"].status())


@app.route("/effects/live/start", methods=["POST"])
def live_start():
    data = request.get_json(silent=True) or {}
    try:
        input_dev  = int(data["input"])  if data.get("input")  not in (None, "") else None
        output_dev = int(data["output"]) if data.get("output") not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "input/output invalides"}), 400
    if input_dev is None or output_dev is None:
        return jsonify({"ok": False, "error": "Sélectionne un micro et une sortie"}), 400
    latency = (data.get("latency") or "low").lower()
    # Apply incoming config (if any) before start so the first chunk is correct.
    if "config" in data:
        services["live"].update_config(_clean_effects(data["config"]) or {})
    ok, err = services["live"].start(input_dev, output_dev, latency=latency)
    return jsonify({"ok": ok, "error": err, "status": services["live"].status()})


@app.route("/effects/live/stop", methods=["POST"])
def live_stop():
    services["live"].stop()
    return jsonify({"ok": True, "status": services["live"].status()})


@app.route("/effects/live/config", methods=["POST"])
def live_config():
    data = request.get_json(silent=True) or {}
    cfg = _clean_effects(data.get("config") or data) or {}
    services["live"].update_config(cfg)
    return jsonify({"ok": True, "config": cfg})


# ─────────────────────────── Voice changer (RVC offline) ──────────────────

import threading as _threading
import uuid as _uuid
import time as _time
_vc_jobs: dict = {}
_vc_jobs_lock = _threading.Lock()


@app.route("/voice-ai/status")
def vc_status():
    return jsonify(services["voice_changer"].status())


@app.route("/voice-ai/voices")
def vc_voices():
    return jsonify({"items": services["voice_changer"].list_voices()})


@app.route("/voice-ai/voices/import", methods=["POST"])
def vc_voice_import():
    pth = request.files.get("pth")
    idx = request.files.get("index")  # optional
    name = (request.form.get("name") or "").strip()
    if not pth or not pth.filename:
        return jsonify({"ok": False, "error": ".pth manquant"}), 400
    if not name:
        name = Path(pth.filename).stem
    from werkzeug.utils import secure_filename
    tmp = Path(services["library"].root) / ".vc-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    pth_path = tmp / secure_filename(pth.filename)
    pth.save(pth_path)
    idx_path = None
    if idx and idx.filename:
        idx_path = tmp / secure_filename(idx.filename)
        idx.save(idx_path)
    try:
        ok, err = services["voice_changer"].import_voice(name, pth_path, idx_path)
    finally:
        try: pth_path.unlink()
        except Exception: pass
        if idx_path:
            try: idx_path.unlink()
            except Exception: pass
    return jsonify({"ok": ok, "error": err, "items": services["voice_changer"].list_voices()})


@app.route("/voice-ai/voices/<name>", methods=["DELETE"])
def vc_voice_delete(name):
    ok = services["voice_changer"].delete_voice(name)
    return jsonify({"ok": ok, "items": services["voice_changer"].list_voices()})


@app.route("/voice-ai/jobs", methods=["POST"])
def vc_jobs_create():
    data = request.get_json(silent=True) or {}
    voice = (data.get("voice") or "").strip()
    source_filename = (data.get("source") or "").strip()  # in soundboard library
    out_name = (data.get("out_name") or "").strip()
    if not voice or not source_filename:
        return jsonify({"ok": False, "error": "voice et source requis"}), 400
    src_path = services["library"].get_path(source_filename)
    if not src_path:
        return jsonify({"ok": False, "error": "Source introuvable"}), 404
    if not out_name:
        out_name = f"{Path(source_filename).stem}-{voice}"
    params = {
        "f0_up_key":   int(data.get("f0_up_key") or 0),
        "f0_method":   str(data.get("f0_method") or "rmvpe"),
        "index_rate":  float(data.get("index_rate") or 0.5),
        "protect":     float(data.get("protect") or 0.33),
    }

    job_id = _uuid.uuid4().hex[:12]
    out_path = Path(services["library"].root) / f"{out_name}.wav"
    n = 2
    while out_path.exists():
        out_path = Path(services["library"].root) / f"{out_name}-{n}.wav"
        n += 1

    with _vc_jobs_lock:
        _vc_jobs[job_id] = {
            "id": job_id, "voice": voice, "source": source_filename,
            "status": "queued", "started": _time.time(),
            "out_filename": out_path.name, "out_path": str(out_path),
            "error": None, "elapsed_ms": 0,
        }

    def runner():
        with _vc_jobs_lock:
            _vc_jobs[job_id]["status"] = "running"
        t0 = _time.monotonic()
        ok, err = services["voice_changer"].convert_file(
            voice_name=voice,
            input_path=str(src_path),
            output_path=str(out_path),
            **params,
        )
        with _vc_jobs_lock:
            j = _vc_jobs[job_id]
            j["elapsed_ms"] = int((_time.monotonic() - t0) * 1000)
            if ok and out_path.exists():
                j["status"] = "done"
            else:
                j["status"] = "error"
                j["error"] = err or "Sortie introuvable"
                try: out_path.unlink()
                except Exception: pass

    _threading.Thread(target=runner, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/voice-ai/jobs/<job_id>")
def vc_jobs_status(job_id):
    with _vc_jobs_lock:
        j = _vc_jobs.get(job_id)
    if not j:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(j)


# Live RVC on the mic
@app.route("/voice-ai/live/status")
def vc_live_status():
    return jsonify(services["live_rvc"].status())


@app.route("/voice-ai/live/start", methods=["POST"])
def vc_live_start():
    data = request.get_json(silent=True) or {}
    try:
        input_dev  = int(data["input"])  if data.get("input")  not in (None, "") else None
        output_dev = int(data["output"]) if data.get("output") not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "input/output invalides"}), 400
    voice = (data.get("voice") or "").strip()
    if not input_dev or not output_dev or not voice:
        return jsonify({"ok": False, "error": "micro, sortie et voix requis"}), 400
    params = {
        "f0_up_key":   int(data.get("f0_up_key") or 0),
        "f0_method":   str(data.get("f0_method") or "rmvpe"),
        "index_rate":  float(data.get("index_rate") or 0.5),
        "protect":     float(data.get("protect") or 0.33),
    }
    latency = (data.get("latency") or "medium").lower()
    ok, err = services["live_rvc"].start(input_dev, output_dev, voice, latency=latency, **params)
    return jsonify({"ok": ok, "error": err, "status": services["live_rvc"].status()})


@app.route("/voice-ai/live/stop", methods=["POST"])
def vc_live_stop():
    services["live_rvc"].stop()
    return jsonify({"ok": True})


@app.route("/voice-ai/live/params", methods=["POST"])
def vc_live_params():
    data = request.get_json(silent=True) or {}
    services["live_rvc"].update_params(**{
        k: data[k] for k in ("f0_up_key", "f0_method", "index_rate", "protect", "rms_mix_rate")
        if k in data
    })
    return jsonify({"ok": True})


@app.route("/effects/file/process", methods=["POST"])
def fx_file_process():
    """Apply an effects chain offline to an uploaded audio file. Stores the
    result in the library under `out_name` (or a sanitized auto name)."""
    import json
    from werkzeug.utils import secure_filename
    import io
    import soundfile as sf
    from effects import apply_chain

    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"ok": False, "error": "Fichier manquant"}), 400
    cfg_raw = request.form.get("config", "{}")
    try:
        cfg = _clean_effects(json.loads(cfg_raw)) or {}
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Config invalide"}), 400
    out_name = (request.form.get("name") or "").strip()
    if not out_name:
        base = secure_filename(f.filename).rsplit(".", 1)[0] or "fx"
        out_name = f"{base}-fx"

    # Decode through the AudioEngine to reuse its ffmpeg fallback.
    tmp_dir = Path(services["library"].root) / ".fx-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = tmp_dir / secure_filename(f.filename)
    f.save(src)
    try:
        data, sr, _ch = services["audio"]._decode(str(src))
        processed = apply_chain(data, sr, cfg)
        # Save as .wav so soundfile can write any sample rate / channel count.
        dest = Path(services["library"].root) / f"{out_name}.wav"
        # If a file with that name exists, suffix with -2, -3, etc.
        n = 2
        while dest.exists():
            dest = Path(services["library"].root) / f"{out_name}-{n}.wav"
            n += 1
        sf.write(str(dest), processed, sr, subtype="PCM_16")
    except Exception as e:
        return jsonify({"ok": False, "error": f"Traitement échoué: {e}"}), 500
    finally:
        try: src.unlink()
        except Exception: pass
    return jsonify({"ok": True, "filename": dest.name})


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


_FX_NUMERIC = {
    "pitch_semitones":  (-24.0, 24.0),
    "lowpass_hz":       (40.0,  20000.0),
    "highpass_hz":      (20.0,  10000.0),
    "robot_hz":         (1.0,   2000.0),
    "tremolo_hz":       (0.1,   30.0),
    "tremolo_depth":    (0.0,   1.0),
    "distortion":       (0.0,   1.0),
    "echo_ms":          (10.0,  2000.0),
    "echo_feedback":    (0.0,   0.95),
    "echo_mix":         (0.0,   1.0),
    "reverb":           (0.0,   1.0),
}


def _clean_effects(fx) -> dict | None:
    if not isinstance(fx, dict):
        return None
    out: dict = {}
    for k, (lo, hi) in _FX_NUMERIC.items():
        if k not in fx:
            continue
        try:
            v = float(fx[k])
        except (TypeError, ValueError):
            continue
        v = max(lo, min(hi, v))
        # Drop keys that are functionally off so we don't waste CPU on no-ops.
        if k == "pitch_semitones" and abs(v) < 0.05:
            continue
        if k in ("distortion", "reverb", "tremolo_depth", "echo_mix") and v < 0.01:
            continue
        out[k] = v
    if fx.get("telephone"):
        out["telephone"] = True
    return out or None


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
    if "effects" in data:
        patch["effects"] = _clean_effects(data["effects"])
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
        cfg = sett.sound(filename)
        per_vol = float(cfg.get("volume", 1.0))
        audio.play(
            str(path),
            device_main=sett.get("output_main"),
            device_monitor=sett.get("output_monitor"),
            per_sound_gain=per_vol,
            monitor_enabled=bool(sett.get("monitor_enabled", True)),
            name=Path(filename).stem,
            effects=cfg.get("effects") or None,
        )  # tuple result ignored — hotkeys are fire-and-forget
    return cb


def rebind_hotkeys():
    """Public entry point — called from main.py on startup and on library change."""
    _rebind_hotkeys()
