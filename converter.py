"""YouTube → audio converter.

Supports re-encoding (mp3/wav/m4a/opus + quality) and a stream-copy mode
("Original") that downloads the native audio with zero re-encoding.
"""
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "opus"}
SUPPORTED_QUALITIES = {"128", "192", "320", "best"}
HISTORY_MAX = 50


def parse_time(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    parts = t.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return t


def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name or "").strip() or "output"


def best_thumbnail(info: dict) -> str | None:
    if info.get("thumbnail"):
        return info["thumbnail"]
    thumbs = info.get("thumbnails") or []
    if not thumbs:
        return None
    return sorted(
        thumbs,
        key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
    )[-1].get("url")


def download_bytes(url: str, timeout: float = 8.0) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def embed_mp3_tags(path: Path, title: str, artist: str, cover_bytes: bytes | None):
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
            tags["APIC"] = APIC(
                encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover_bytes
            )
        tags.save(path, v2_version=3)
    except Exception:
        pass


def embed_m4a_tags(path: Path, title: str, artist: str, cover_bytes: bytes | None):
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


def embed_metadata(path: Path, fmt: str, title: str, artist: str, thumbnail_url: str | None):
    cover = download_bytes(thumbnail_url) if thumbnail_url else None
    if fmt == "mp3":
        embed_mp3_tags(path, title, artist, cover)
    elif fmt == "m4a":
        embed_m4a_tags(path, title, artist, cover)


class Converter:
    """Manages YT downloads, history, and the job queue."""

    def __init__(self, downloads_dir: Path, ffmpeg_cmd: str = "ffmpeg"):
        self.downloads = Path(downloads_dir)
        self.downloads.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = ffmpeg_cmd
        self.history_file = self.downloads / ".history.json"
        self.jobs: dict[str, dict] = {}
        self._hist_lock = threading.Lock()
        self.on_history_change = None
        self.on_library_change = None

    # ------------------------ History --------------------------------------

    def read_history(self) -> list[dict]:
        if not self.history_file.exists():
            return []
        try:
            return json.loads(self.history_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def append_history(self, entry: dict):
        with self._hist_lock:
            items = self.read_history()
            items.insert(0, entry)
            items = items[:HISTORY_MAX]
            try:
                self.history_file.write_text(
                    json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
        if self.on_history_change:
            try:
                self.on_history_change()
            except Exception:
                pass
        if self.on_library_change:
            try:
                self.on_library_change()
            except Exception:
                pass

    def clear_history(self):
        with self._hist_lock:
            try:
                self.history_file.unlink(missing_ok=True)
            except Exception:
                pass
        if self.on_history_change:
            try:
                self.on_history_change()
            except Exception:
                pass

    # ------------------------ yt-dlp helpers -------------------------------

    def _ffmpeg_dir(self) -> str | None:
        p = Path(self.ffmpeg)
        return str(p.parent) if p.is_absolute() else None

    def _ydl_base_opts(self) -> dict:
        opts = {"quiet": True, "no_warnings": True, "noplaylist": False}
        d = self._ffmpeg_dir()
        if d:
            opts["ffmpeg_location"] = d
        return opts

    def fetch_info(self, url: str) -> dict:
        import yt_dlp
        opts = self._ydl_base_opts()
        opts["extract_flat"] = "in_playlist"
        with yt_dlp.YoutubeDL(opts) as ydl:
            data = ydl.extract_info(url, download=False)

        if data.get("_type") == "playlist":
            entries = [e for e in (data.get("entries") or []) if e]
            preview = entries[:6]
            return {
                "type": "playlist",
                "title": data.get("title") or "Playlist",
                "uploader": data.get("uploader") or "",
                "count": len(entries),
                "thumbnail": best_thumbnail(data)
                or (preview[0].get("thumbnail") if preview else None),
                "items": [
                    {
                        "title": e.get("title"),
                        "duration": e.get("duration"),
                        "thumbnail": e.get("thumbnail"),
                    }
                    for e in preview
                ],
            }

        return {
            "type": "video",
            "title": data.get("title"),
            "uploader": data.get("uploader") or data.get("channel") or "",
            "duration": data.get("duration"),
            "thumbnail": best_thumbnail(data),
            "view_count": data.get("view_count"),
        }

    # ------------------------ Job submission -------------------------------

    def submit(
        self,
        url: str,
        start: str = "",
        end: str = "",
        name: str = "",
        fmt: str = "mp3",
        quality: str = "320",
        original: bool = False,
    ) -> str:
        fmt = fmt.lower() if not original else "original"
        if not original and fmt not in SUPPORTED_FORMATS:
            fmt = "mp3"
        if quality not in SUPPORTED_QUALITIES:
            quality = "320"

        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "En attente",
            "progress": 0.0,
            "file": None,
            "error": None,
            "format": fmt,
        }
        threading.Thread(
            target=self._run,
            args=(job_id, url, start, end, name, fmt, quality, original),
            daemon=True,
        ).start()
        return job_id

    def job_status(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "status": job["status"],
            "progress": job.get("progress", 0),
            "speed": job.get("speed"),
            "current_item": job.get("current_item"),
            "total_items": job.get("total_items"),
            "filename": job.get("filename"),
            "error": job.get("error"),
            "format": job.get("format"),
        }

    def job_file(self, job_id: str) -> Path | None:
        job = self.jobs.get(job_id)
        if not job or job.get("status") != "done":
            return None
        return Path(job["file"]) if job.get("file") else None

    # ------------------------ Worker ---------------------------------------

    def _progress_hook(self, job: dict, item_index: int = 0, total_items: int = 1):
        def hook(d: dict):
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                done = d.get("downloaded_bytes") or 0
                if total > 0:
                    pct = (done / total) * 100.0
                else:
                    s = (d.get("_percent_str") or "0%").strip().rstrip("%")
                    try:
                        pct = float(s)
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
                    job["speed"] = (
                        f"{speed / 1_000_000:.1f} MB/s"
                        if speed > 1_000_000
                        else f"{speed / 1000:.0f} KB/s"
                    )
            elif d["status"] == "finished":
                job["status"] = "Conversion audio"
        return hook

    def _build_opts_encode(self, tmp_dir: Path, fmt: str, quality: str, hook) -> dict:
        ffmpeg_quality = "0" if quality == "best" else quality
        postprocessor = {"key": "FFmpegExtractAudio", "preferredcodec": fmt}
        if fmt in ("mp3", "m4a", "opus"):
            postprocessor["preferredquality"] = ffmpeg_quality

        opts = {
            "format": "bestaudio/best",
            "postprocessors": [postprocessor],
            "outtmpl": str(tmp_dir / "%(autonumber)03d_%(title).80s.%(ext)s"),
            "noplaylist": True,
            "progress_hooks": [hook],
            "quiet": True,
            "no_warnings": True,
        }
        d = self._ffmpeg_dir()
        if d:
            opts["ffmpeg_location"] = d
        return opts

    def _build_opts_original(self, tmp_dir: Path, hook) -> dict:
        # No postprocessor → stream copy of the native audio (typically .webm/.opus or .m4a)
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(tmp_dir / "%(autonumber)03d_%(title).80s.%(ext)s"),
            "noplaylist": True,
            "progress_hooks": [hook],
            "quiet": True,
            "no_warnings": True,
        }
        d = self._ffmpeg_dir()
        if d:
            opts["ffmpeg_location"] = d
        return opts

    def _process_single(
        self,
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
        original: bool,
    ) -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            if original:
                opts = self._build_opts_original(
                    tmp_dir, self._progress_hook(job, item_index, total_items)
                )
            else:
                opts = self._build_opts_encode(
                    tmp_dir, fmt, quality, self._progress_hook(job, item_index, total_items)
                )

            import yt_dlp
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            if original:
                produced = sorted(tmp_dir.iterdir())
            else:
                produced = sorted(tmp_dir.glob(f"*.{fmt}")) or sorted(tmp_dir.iterdir())
            if not produced:
                raise RuntimeError("Fichier audio introuvable après téléchargement.")
            src = produced[0]
            out_ext = src.suffix.lstrip(".").lower() if original else fmt

            title = info.get("title") or src.stem
            artist = info.get("uploader") or info.get("channel") or ""
            thumb = best_thumbnail(info)
            base_name = sanitize(name_override or title)
            dest = self.downloads / f"{base_name}.{out_ext}"
            n = 1
            while dest.exists():
                dest = self.downloads / f"{base_name} ({n}).{out_ext}"
                n += 1

            if start or end:
                job["status"] = "Découpage"
                ff_cmd = [self.ffmpeg, "-y", "-i", str(src)]
                if start:
                    ff_cmd += ["-ss", parse_time(start)]
                if end:
                    ff_cmd += ["-to", parse_time(end)]
                ff_cmd += ["-acodec", "copy", str(dest)]
                r = subprocess.run(ff_cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-600:])
            else:
                shutil.move(str(src), str(dest))

            if not original:
                job["status"] = "Métadonnées"
                embed_metadata(dest, out_ext, title, artist, thumb)

            self.append_history({
                "id": uuid.uuid4().hex,
                "title": title,
                "artist": artist,
                "thumbnail": thumb,
                "filename": dest.name,
                "format": out_ext,
                "quality": "original" if original else quality,
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

    def _run(
        self,
        job_id: str,
        url: str,
        start: str,
        end: str,
        name: str,
        fmt: str,
        quality: str,
        original: bool,
    ):
        import yt_dlp

        job = self.jobs[job_id]
        try:
            job["status"] = "Analyse"
            job["progress"] = 0.0

            info_opts = self._ydl_base_opts()
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
                    dest = self._process_single(
                        job, entry_url, entry, i, total, fmt, quality, "", "", None, original
                    )
                    files.append(dest)
                if not files:
                    raise RuntimeError("Aucun morceau n'a pu être téléchargé.")
                if len(files) == 1:
                    job["file"] = str(files[0])
                    job["filename"] = files[0].name
                else:
                    job["status"] = "Compression"
                    base = sanitize(info.get("title") or "playlist")
                    zip_path = self.downloads / f"{base}.zip"
                    n = 1
                    while zip_path.exists():
                        zip_path = self.downloads / f"{base} ({n}).zip"
                        n += 1
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
                        for f in files:
                            z.write(f, arcname=f.name)
                    job["file"] = str(zip_path)
                    job["filename"] = zip_path.name
            else:
                dest = self._process_single(
                    job, url, info, 0, 1, fmt, quality, start, end, name or None, original
                )
                job["file"] = str(dest)
                job["filename"] = dest.name

            job["progress"] = 100.0
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
