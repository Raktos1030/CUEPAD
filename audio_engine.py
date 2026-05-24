"""Audio engine: callback-based multi-output playback with live volume + stop."""
import threading
from pathlib import Path


def _is_virtual_cable(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("vb-audio", "cable", "voicemeeter", "virtual"))


class _Voice:
    """One playing instance, fed by a sounddevice callback."""
    __slots__ = ("data", "pos", "sr", "channels", "channel_type",
                 "per_sound_gain", "stream")

    def __init__(self, data, sr, channels, channel_type, per_sound_gain):
        self.data = data           # 2D ndarray (samples, channels), float32
        self.pos = 0
        self.sr = sr
        self.channels = channels
        self.channel_type = channel_type  # "main" or "monitor"
        self.per_sound_gain = per_sound_gain
        self.stream = None


class AudioEngine:
    def __init__(self, ffmpeg_path: str | None = None):
        self.ffmpeg_path = ffmpeg_path
        self._voices: set[_Voice] = set()
        self._voices_lock = threading.Lock()

        # Live state read by every callback tick.
        self._global_main = 1.0
        self._global_monitor = 0.7
        self._mute_monitor = False

        if ffmpeg_path:
            p = Path(ffmpeg_path)
            if p.is_absolute() and p.exists():
                self.ffmpeg_path = str(p)

    # ─── Live state setters (called from /settings) ────────────────────────
    def set_global_main(self, v: float):
        self._global_main = max(0.0, min(2.0, float(v)))

    def set_global_monitor(self, v: float):
        self._global_monitor = max(0.0, min(2.0, float(v)))

    def set_monitor_muted(self, muted: bool):
        self._mute_monitor = bool(muted)

    @staticmethod
    def list_devices() -> dict:
        try:
            import sounddevice as sd
        except Exception as e:
            return {"outputs": [], "error": f"sounddevice unavailable: {e}"}

        try:
            devs = sd.query_devices()
        except Exception as e:
            return {"outputs": [], "error": str(e)}

        default = sd.default.device
        default_out = default[1] if isinstance(default, (list, tuple)) else default

        outputs = []
        for i, d in enumerate(devs):
            if d.get("max_output_channels", 0) <= 0:
                continue
            name = d.get("name", "")
            outputs.append({
                "id": i,
                "name": name,
                "default": i == default_out,
                "virtual": _is_virtual_cable(name),
                "channels": d.get("max_output_channels", 0),
                "samplerate": int(d.get("default_samplerate", 44100)),
            })
        return {"outputs": outputs}

    def play(
        self,
        file_path: str,
        device_main,
        device_monitor,
        per_sound_gain: float,
        monitor_enabled: bool,
    ) -> tuple[bool, str | None]:
        try:
            data, sr, channels = self._decode(file_path)
        except Exception as e:
            return False, f"Décodage impossible: {e}"

        import numpy as np
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            channels = 1
        # Ensure contiguous float32 for sounddevice callback slicing
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        played = False
        if device_main is not None:
            if self._start_voice(data, sr, channels, device_main,
                                 per_sound_gain, "main"):
                played = True

        if monitor_enabled and device_monitor is not None and device_monitor != device_main:
            if self._start_voice(data, sr, channels, device_monitor,
                                 per_sound_gain, "monitor"):
                played = True

        # Fallback: no devices configured → at least play on default output.
        if not played and monitor_enabled:
            if self._start_voice(data, sr, channels, None,
                                 per_sound_gain, "monitor"):
                played = True

        return played, None if played else "Aucune sortie audio configurée"

    def _start_voice(self, data, sr, channels, device,
                     per_sound_gain, channel_type) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
        except Exception:
            return False

        voice = _Voice(data, sr, channels, channel_type, per_sound_gain)

        def callback(outdata, frames, time_info, status):
            n_src = voice.data.shape[0] - voice.pos
            n = min(frames, n_src)
            if channel_type == "main":
                g = self._global_main * voice.per_sound_gain
            else:
                g = 0.0 if self._mute_monitor else (
                    self._global_monitor * voice.per_sound_gain
                )

            if n > 0:
                chunk = voice.data[voice.pos:voice.pos + n] * g
                out_ch = outdata.shape[1]
                src_ch = chunk.shape[1]
                if src_ch == out_ch:
                    outdata[:n] = chunk
                elif src_ch == 1:
                    outdata[:n] = np.broadcast_to(chunk, (n, out_ch))
                elif out_ch == 1:
                    outdata[:n] = chunk.mean(axis=1, keepdims=True)
                else:
                    outdata[:n] = 0
                    m = min(src_ch, out_ch)
                    outdata[:n, :m] = chunk[:, :m]
                voice.pos += n

            if n < frames:
                outdata[n:] = 0
                raise sd.CallbackStop()

        def finished():
            with self._voices_lock:
                self._voices.discard(voice)

        try:
            info = (
                sd.query_devices(device, "output") if device is not None
                else sd.query_devices(kind="output")
            )
            dev_ch = int(info.get("max_output_channels") or channels)
            out_channels = max(1, min(channels, dev_ch)) if channels > 1 else min(2, dev_ch) or 1
        except Exception:
            out_channels = channels

        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=out_channels,
                device=device,
                dtype="float32",
                callback=callback,
                finished_callback=finished,
            )
        except Exception:
            return False

        voice.stream = stream
        with self._voices_lock:
            self._voices.add(voice)
        try:
            stream.start()
            return True
        except Exception:
            with self._voices_lock:
                self._voices.discard(voice)
            try:
                stream.close()
            except Exception:
                pass
            return False

    def stop_all(self):
        with self._voices_lock:
            voices = list(self._voices)
            self._voices.clear()
        for v in voices:
            try:
                v.stream.stop()
            except Exception:
                pass
            try:
                v.stream.close()
            except Exception:
                pass

    def _decode(self, file_path: str):
        # Try soundfile first — handles wav/flac/ogg natively (and mp3 with
        # libsndfile >= 1.1) without needing an ffmpeg binary on PATH.
        try:
            import soundfile as sf
            data, sr = sf.read(file_path, dtype="float32", always_2d=True)
            channels = data.shape[1]
            return data, int(sr), int(channels)
        except Exception as sf_err:
            sf_msg = str(sf_err)

        # Fallback: shell out to ffmpeg ourselves (m4a/opus/aac/webm and
        # old libsndfile). We used to go through pydub here, but pydub's
        # `from_file` calls ffprobe even with format=hint — and ffprobe.exe
        # isn't always bundled. Direct ffmpeg → wav → soundfile skips that.
        import io
        import subprocess
        ext = Path(file_path).suffix.lstrip(".").lower() or "?"
        ffmpeg = self.ffmpeg_path or "ffmpeg"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-i", file_path, "-vn", "-f", "wav", "-"],
                capture_output=True,
            )
        except FileNotFoundError as ff_err:
            raise RuntimeError(
                f"FFmpeg introuvable — requis pour décoder .{ext}. "
                f"Réinstallez Q-Pad ou ajoutez ffmpeg au PATH système. "
                f"(testé: {ffmpeg!r}; soundfile: {sf_msg})"
            ) from ff_err
        if proc.returncode != 0:
            tail = proc.stderr[-400:].decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg a renvoyé {proc.returncode}: {tail} "
                f"(soundfile: {sf_msg})"
            )
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
            return data, int(sr), int(data.shape[1])
        except Exception as sf2:
            raise RuntimeError(
                f"Lecture WAV depuis ffmpeg a échoué: {sf2} "
                f"(soundfile initial: {sf_msg})"
            )
