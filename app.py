import os
import re
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

DOWNLOADS = Path("downloads")
DOWNLOADS.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


def _parse_time(t: str) -> str:
    """Accept 1:30, 01:30, 00:01:30 — returns ffmpeg-compatible string."""
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


def _run_job(job_id: str, url: str, start: str, end: str, out_name: str):
    job = jobs[job_id]
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # 1. Download best audio
        job["status"] = "Téléchargement en cours…"
        tmp_audio = tmp_dir / "audio.%(ext)s"
        dl_cmd = [
            "yt-dlp",
            "--format", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", str(tmp_audio),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(dl_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            job["status"] = "error"
            job["error"] = result.stderr[-500:]
            return

        # Find downloaded file
        downloaded = list(tmp_dir.glob("*.mp3"))
        if not downloaded:
            downloaded = list(tmp_dir.glob("*.*"))
        if not downloaded:
            job["status"] = "error"
            job["error"] = "Fichier audio introuvable après téléchargement."
            return

        src = downloaded[0]
        dest = DOWNLOADS / f"{out_name}_{job_id[:8]}.mp3"

        # 2. Cut with ffmpeg if timestamps given
        if start or end:
            job["status"] = "Découpage en cours…"
            ff_cmd = ["ffmpeg", "-y", "-i", str(src)]
            if start:
                ff_cmd += ["-ss", _parse_time(start)]
            if end:
                ff_cmd += ["-to", _parse_time(end)]
            ff_cmd += ["-acodec", "copy", str(dest)]
            result = subprocess.run(ff_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                job["status"] = "error"
                job["error"] = result.stderr[-500:]
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
        # Cleanup tmp
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

    thread = threading.Thread(target=_run_job, args=(job_id, url, start, end, out_name), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify({
        "status": job["status"],
        "filename": job.get("filename"),
        "error": job.get("error"),
    })


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Fichier non prêt", 404
    path = job["file"]
    return send_file(path, as_attachment=True, download_name=job["filename"])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
