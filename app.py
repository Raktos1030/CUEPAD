import re
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# --- Config (overridden by main.py before Flask starts) ---
FFMPEG_CMD = "ffmpeg"
DOWNLOADS = Path("downloads")

if getattr(sys, "frozen", False):
    _template_dir = Path(sys._MEIPASS) / "templates"
else:
    _template_dir = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(_template_dir))
jobs: dict[str, dict] = {}


def configure(ffmpeg_path: str | None = None, downloads_dir: Path | None = None):
    global FFMPEG_CMD, DOWNLOADS
    if ffmpeg_path:
        FFMPEG_CMD = ffmpeg_path
    if downloads_dir:
        DOWNLOADS = Path(downloads_dir)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)


def _parse_time(t: str) -> str:
    t = t.strip()
    if not t:
        return ""
    parts = t.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return t


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip() or "output"


def _progress_hook(job: dict):
    def hook(d: dict):
        if d["status"] == "downloading":
            pct = d.get("_percent_str", "").strip()
            job["status"] = f"Téléchargement {pct}…" if pct else "Téléchargement en cours…"
        elif d["status"] == "finished":
            job["status"] = "Conversion audio…"
    return hook


def _run_job(job_id: str, url: str, start: str, end: str, out_name: str):
    import yt_dlp
    import subprocess

    job = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        job["status"] = "Téléchargement en cours…"

        ffmpeg_dir = str(Path(FFMPEG_CMD).parent) if Path(FFMPEG_CMD).is_absolute() else None

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}],
            "outtmpl": str(tmp_dir / "audio.%(ext)s"),
            "noplaylist": True,
            "progress_hooks": [_progress_hook(job)],
            "quiet": True,
            "no_warnings": True,
        }
        if ffmpeg_dir:
            ydl_opts["ffmpeg_location"] = ffmpeg_dir

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        files = list(tmp_dir.glob("*.mp3")) or list(tmp_dir.glob("*.*"))
        if not files:
            job["status"] = "error"
            job["error"] = "Fichier audio introuvable après téléchargement."
            return

        src = files[0]
        dest = DOWNLOADS / f"{out_name}_{job_id[:8]}.mp3"

        if start or end:
            job["status"] = "Découpage en cours…"
            ff_cmd = [FFMPEG_CMD, "-y", "-i", str(src)]
            if start:
                ff_cmd += ["-ss", _parse_time(start)]
            if end:
                ff_cmd += ["-to", _parse_time(end)]
            ff_cmd += ["-acodec", "copy", str(dest)]
            r = subprocess.run(ff_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                job["status"] = "error"
                job["error"] = r.stderr[-600:]
                return
        else:
            src.rename(dest)

        job["file"] = str(dest)
        job["filename"] = dest.name
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        for f in tmp_dir.iterdir():
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    url = request.form.get("url", "").strip()
    start = request.form.get("start", "").strip()
    end = request.form.get("end", "").strip()
    out_name = _sanitize(request.form.get("name", "output"))

    if not url:
        return jsonify({"error": "URL manquante"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "En attente…", "file": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, url, start, end, out_name), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify({"status": job["status"], "filename": job.get("filename"), "error": job.get("error")})


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Fichier non prêt", 404
    return send_file(job["file"], as_attachment=True, download_name=job["filename"])


if __name__ == "__main__":
    configure()
    app.run(debug=True, port=5000)
