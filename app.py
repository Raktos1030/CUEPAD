import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from flask import Flask, jsonify, render_template, request, send_file

# --- Config (overridden by main.py before Flask starts) ---
FFMPEG_CMD = "ffmpeg"
DOWNLOADS = Path("downloads")
HISTORY_FILE = Path("history.json")
HISTORY_MAX = 50

if getattr(sys, "frozen", False):
    _template_dir = Path(sys._MEIPASS) / "templates"
else:
    _template_dir = Path(__file__).parent / "templates"

app = Flask(__name__, template_folder=str(_template_dir))
jobs: dict[str, dict] = {}
_history_lock = threading.Lock()

SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "opus"}
SUPPORTED_QUALITIES = {"128", "192", "320", "best"}


def configure(ffmpeg_path: str | None = None, downloads_dir: Path | None = None):
    global FFMPEG_CMD, DOWNLOADS, HISTORY_FILE
    if ffmpeg_path:
        FFMPEG_CMD = ffmpeg_path
    if downloads_dir:
        DOWNLOADS = Path(downloads_dir)
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE = DOWNLOADS / ".history.json"


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


def _ffmpeg_dir() -> str | None:
    return str(Path(FFMPEG_CMD).parent) if Path(FFMPEG_CMD).is_absolute() else None


def _ydl_base_opts() -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": False}
    d = _ffmpeg_dir()
    if d:
        opts["ffmpeg_location"] = d
    return opts


def _best_thumbnail(info: dict) -> str | None:
    if info.get("thumbnail"):
        return info["thumbnail"]
    thumbs = info.get("thumbnails") or []
    if not thumbs:
        return None
    return sorted(thumbs, key=lambda t: (t.get("width") or 0) * (t.get("height") or 0))[-1].get("url")


def _read_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_history(entry: dict):
    with _history_lock:
        items = _read_history()
        items.insert(0, entry)
        items = items[:HISTORY_MAX]
        try:
            HISTORY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def _download_bytes(url: str, timeout: float = 8.0) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _embed_mp3_tags(path: Path, title: str, artist: str, cover_bytes: bytes | None):
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError, TIT2, TPE1, APIC
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        if title:
            tags["TIT2"] = TIT2(encoding=3, text=title)
        if artist:
            tags["TPE1"] = TPE1(encoding=3, text=artist)
        if cover_bytes:
            tags["APIC"] = APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover_bytes)
        tags.save(path, v2_version=3)
    except Exception:
        pass


def _embed_m4a_tags(path: Path, title: str, artist: str, cover_bytes: bytes | None):
    try:
        from mutagen.mp4 import MP4, MP4Cover
        audio = MP4(path)
        if title:
            audio["\xa9nam"] = [title]
        if artist:
            audio["\xa9ART"] = [artist]
        if cover_bytes:
            audio["covr"] = [MP4Cover(cover_bytes, imageformat=MP4Cover.FORMAT_JPEG)]
        audio.save()
    except Exception:
        pass


def _embed_metadata(path: Path, fmt: str, title: str, artist: str, thumbnail_url: str | None):
    cover = _download_bytes(thumbnail_url) if thumbnail_url else None
    if fmt == "mp3":
        _embed_mp3_tags(path, title, artist, cover)
    elif fmt == "m4a":
        _embed_m4a_tags(path, title, artist, cover)


def _progress_hook(job: dict, item_index: int = 0, total_items: int = 1):
    def hook(d: dict):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total > 0:
                pct = (done / total) * 100.0
            else:
                pct_str = (d.get("_percent_str") or "0%").strip().rstrip("%")
                try:
                    pct = float(pct_str)
                except ValueError:
                    pct = 0.0
            if total_items > 1:
                overall = ((item_index + pct / 100.0) / total_items) * 100.0
                job["progress"] = round(overall, 1)
                job["status"] = f"Téléchargement ({item_index + 1}/{total_items})"
            else:
                job["progress"] = round(pct, 1)
                job["status"] = "Téléchargement"
            speed = d.get("speed")
            if speed:
                job["speed"] = f"{speed / 1_000_000:.1f} MB/s" if speed > 1_000_000 else f"{speed / 1000:.0f} KB/s"
        elif d["status"] == "finished":
            job["status"] = "Conversion audio"
    return hook


def _build_ydl_opts(tmp_dir: Path, fmt: str, quality: str, hook):
    if quality == "best":
        ffmpeg_quality = "0"
    else:
        ffmpeg_quality = quality

    postprocessor = {"key": "FFmpegExtractAudio", "preferredcodec": fmt}
    if fmt in ("mp3", "m4a", "opus"):
        postprocessor["preferredquality"] = ffmpeg_quality

    opts = {
        "format": "bestaudio/best",
        "postprocessors": [postprocessor],
        "outtmpl": str(tmp_dir / "%(autonumber)03d_%(title).80s.%(ext)s"),
        "noplaylist": False,
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
    }
    d = _ffmpeg_dir()
    if d:
        opts["ffmpeg_location"] = d
    return opts


def _process_single(
    job: dict,
    url: str,
    info: dict,
    item_index: int,
    total_items: int,
    fmt: str,
    quality: str,
    start: str,
    end: str,
    name_override: str | None,
):
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        opts = _build_ydl_opts(tmp_dir, fmt, quality, _progress_hook(job, item_index, total_items))
        opts["noplaylist"] = True
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        produced = sorted(tmp_dir.glob(f"*.{fmt}")) or sorted(tmp_dir.iterdir())
        if not produced:
            raise RuntimeError("Fichier audio introuvable après téléchargement.")
        src = produced[0]

        title = info.get("title") or src.stem
        artist = info.get("uploader") or info.get("channel") or ""
        thumb = _best_thumbnail(info)
        base_name = _sanitize(name_override or title)
        dest = DOWNLOADS / f"{base_name}.{fmt}"
        n = 1
        while dest.exists():
            dest = DOWNLOADS / f"{base_name} ({n}).{fmt}"
            n += 1

        if start or end:
            job["status"] = "Découpage"
            ff_cmd = [FFMPEG_CMD, "-y", "-i", str(src)]
            if start:
                ff_cmd += ["-ss", _parse_time(start)]
            if end:
                ff_cmd += ["-to", _parse_time(end)]
            ff_cmd += ["-acodec", "copy", str(dest)]
            r = subprocess.run(ff_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-600:])
        else:
            shutil.move(str(src), str(dest))

        job["status"] = "Métadonnées"
        _embed_metadata(dest, fmt, title, artist, thumb)

        _append_history({
            "id": uuid.uuid4().hex,
            "title": title,
            "artist": artist,
            "thumbnail": thumb,
            "filename": dest.name,
            "format": fmt,
            "quality": quality,
            "size": dest.stat().st_size,
            "ts": int(time.time()),
            "url": url,
        })
        return dest
    finally:
        for f in tmp_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            tmp_dir.rmdir()
        except Exception:
            pass


def _run_job(job_id: str, url: str, start: str, end: str, name: str, fmt: str, quality: str):
    import yt_dlp

    job = jobs[job_id]
    try:
        job["status"] = "Analyse"
        job["progress"] = 0.0

        info_opts = _ydl_base_opts()
        info_opts["extract_flat"] = False
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = info.get("entries") if info.get("_type") == "playlist" else None
        files: list[Path] = []

        if entries:
            entries = [e for e in entries if e]
            total = len(entries)
            job["total_items"] = total
            for i, entry in enumerate(entries):
                job["current_item"] = i + 1
                entry_url = entry.get("webpage_url") or entry.get("url")
                if not entry_url:
                    continue
                dest = _process_single(
                    job, entry_url, entry, i, total, fmt, quality, "", "", None,
                )
                files.append(dest)
            if not files:
                raise RuntimeError("Aucun morceau n'a pu être téléchargé.")
            if len(files) == 1:
                job["file"] = str(files[0])
                job["filename"] = files[0].name
            else:
                # Zip them together
                job["status"] = "Compression"
                zip_path = DOWNLOADS / f"{_sanitize(info.get('title') or 'playlist')}.zip"
                n = 1
                while zip_path.exists():
                    zip_path = DOWNLOADS / f"{_sanitize(info.get('title') or 'playlist')} ({n}).zip"
                    n += 1
                import zipfile
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
                    for f in files:
                        z.write(f, arcname=f.name)
                job["file"] = str(zip_path)
                job["filename"] = zip_path.name
        else:
            dest = _process_single(job, url, info, 0, 1, fmt, quality, start, end, name or None)
            job["file"] = str(dest)
            job["filename"] = dest.name

        job["progress"] = 100.0
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info", methods=["POST"])
def info():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400
    try:
        import yt_dlp
        opts = _ydl_base_opts()
        opts["extract_flat"] = "in_playlist"
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)

        if data.get("_type") == "playlist":
            entries = [e for e in (data.get("entries") or []) if e]
            preview = entries[:6]
            return jsonify({
                "type": "playlist",
                "title": data.get("title") or "Playlist",
                "uploader": data.get("uploader") or "",
                "count": len(entries),
                "thumbnail": _best_thumbnail(data) or (preview[0].get("thumbnail") if preview else None),
                "items": [
                    {
                        "title": e.get("title"),
                        "duration": e.get("duration"),
                        "thumbnail": e.get("thumbnail"),
                    }
                    for e in preview
                ],
            })

        return jsonify({
            "type": "video",
            "title": data.get("title"),
            "uploader": data.get("uploader") or data.get("channel") or "",
            "duration": data.get("duration"),
            "thumbnail": _best_thumbnail(data),
            "view_count": data.get("view_count"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/convert", methods=["POST"])
def convert():
    data = request.get_json(silent=True) or request.form
    url = (data.get("url") or "").strip()
    start = (data.get("start") or "").strip()
    end = (data.get("end") or "").strip()
    name = _sanitize(data.get("name") or "") if data.get("name") else ""
    fmt = (data.get("format") or "mp3").lower()
    quality = (data.get("quality") or "320")

    if fmt not in SUPPORTED_FORMATS:
        fmt = "mp3"
    if quality not in SUPPORTED_QUALITIES:
        quality = "320"

    if not url:
        return jsonify({"error": "URL manquante"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "En attente",
        "progress": 0.0,
        "file": None,
        "error": None,
        "format": fmt,
    }
    threading.Thread(
        target=_run_job,
        args=(job_id, url, start, end, name, fmt, quality),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify({
        "status": job["status"],
        "progress": job.get("progress", 0),
        "speed": job.get("speed"),
        "current_item": job.get("current_item"),
        "total_items": job.get("total_items"),
        "filename": job.get("filename"),
        "error": job.get("error"),
        "format": job.get("format"),
    })


@app.route("/download/<job_id>")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Fichier non prêt", 404
    return send_file(job["file"], as_attachment=True, download_name=job["filename"])


@app.route("/history")
def history():
    items = _read_history()
    return jsonify({"items": items, "downloads_dir": str(DOWNLOADS)})


@app.route("/history/clear", methods=["POST"])
def history_clear():
    with _history_lock:
        try:
            HISTORY_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/history/file/<path:filename>")
def history_file(filename: str):
    safe = Path(filename).name
    target = DOWNLOADS / safe
    if not target.exists():
        return "Fichier introuvable", 404
    return send_file(target, as_attachment=True, download_name=safe)


if __name__ == "__main__":
    configure()
    app.run(debug=True, port=5000)
