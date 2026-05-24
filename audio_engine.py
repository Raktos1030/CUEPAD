"""Audio engine: callback-based multi-output playback with pause/seek/speed."""
import io
import subprocess
import threading
import time
import uuid
from pathlib import Path


def _is_virtual_cable(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("vb-audio", "cable", "voicemeeter", "virtual"))


def _time_stretch(data, sr: int, speed: float):
    """Pitch-preserving time stretch. speed > 1 = faster, < 1 = slower."""
    import numpy as np
    if abs(speed - 1.0) < 0.005:
        return data
    try:
        import audiotsm
        from audiotsm.io.array import ArrayReader, ArrayWriter
    except Exception:
        return data  # no stretch lib → return as-is

    if data.ndim == 1:
        data_in = data.reshape(1, -1).astype(np.float32)
        n_ch = 1
    else:
        # audiotsm wants (channels, samples)
        data_in = data.T.astype(np.float32)
        n_ch = data.shape[1]

    reader = ArrayReader(data_in)
    writer = ArrayWriter(n_ch)
    tsm = audiotsm.wsola(n_ch, speed=speed)
    tsm.run(reader, writer)
    out = writer.data.T
    return out.astype(np.float32)


class _Voice:
    """One sounddevice OutputStream tied to a Playback."""
    __slots__ = ("playback", "pos", "channel_type", "stream")

    def __init__(self, playback, channel_type):
        self.playback = playback
        self.pos = 0
        self.channel_type = channel_type
        self.stream = None


class Playback:
    """One user-initiated sound playback. May drive 1 or 2 output streams."""

    def __init__(self, pb_id, name, data, sr, channels, per_sound_gain):
        self.id = pb_id
        self.name = name
        self.original_data = data    # unmodified — used to re-stretch
        self.current_data = data     # what voices read (stretched copy)
        self.sr = int(sr)
        self.channels = int(channels)
        self.per_sound_gain = float(per_sound_gain)
        self.speed = 1.0
        self.paused = False
        self.voices: list[_Voice] = []
        self.start_ts = time.monotonic()

    @property
    def duration_sec(self) -> float:
        return len(self.original_data) / self.sr if self.sr else 0.0

    @property
    def position_sec(self) -> float:
        if not self.voices:
            return 0.0
        cur = max(v.pos for v in self.voices)
        # current_data is stretched by 1/speed relative to original, so
        # real-time elapsed in original timeline = (cur / sr) * speed.
        return min(self.duration_sec, (cur / self.sr) * self.speed)

    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pos_sec": round(self.position_sec, 3),
            "dur_sec": round(self.duration_sec, 3),
            "speed": round(self.speed, 3),
            "paused": self.paused,
            "started_ts": self.start_ts,
        }


class AudioEngine:
    def __init__(self, ffmpeg_path: str | None = None):
        self.ffmpeg_path = ffmpeg_path
        self._playbacks: dict[str, Playback] = {}
        self._lock = threading.Lock()

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

    # ─── Playback API ──────────────────────────────────────────────────────
    def list_playbacks(self) -> list[dict]:
        with self._lock:
            # Sort by start time so the UI shows oldest → newest.
            return [pb.info() for pb in sorted(
                self._playbacks.values(), key=lambda p: p.start_ts)]

    def play(
        self,
        file_path: str,
        device_main,
        device_monitor,
        per_sound_gain: float,
        monitor_enabled: bool,
        name: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        try:
            data, sr, channels = self._decode(file_path)
        except Exception as e:
            return False, f"Décodage impossible: {e}", None

        import numpy as np
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            channels = 1
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        pb_id = uuid.uuid4().hex[:12]
        pb = Playback(pb_id, name or Path(file_path).stem,
                      data, sr, channels, per_sound_gain)

        with self._lock:
            self._playbacks[pb_id] = pb

        played = False
        if self._start_voice(pb, device_main, "main"):
            played = True
        if monitor_enabled and device_monitor is not None and device_monitor != device_main:
            if self._start_voice(pb, device_monitor, "monitor"):
                played = True
        if not played and monitor_enabled:
            # No device configured at all — fall back to system default so the
            # user at least hears something.
            if self._start_voice(pb, None, "monitor"):
                played = True

        if not played:
            with self._lock:
                self._playbacks.pop(pb_id, None)
            return False, "Aucune sortie audio configurée", None
        return True, None, pb_id

    def _start_voice(self, playback: Playback, device, channel_type: str) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
        except Exception:
            return False

        voice = _Voice(playback, channel_type)

        def callback(outdata, frames, time_info, status):
            if playback.paused:
                outdata.fill(0)
                return
            data = playback.current_data
            n_src = data.shape[0] - voice.pos
            n = min(frames, n_src)
            if channel_type == "main":
                g = self._global_main * playback.per_sound_gain
            else:
                g = 0.0 if self._mute_monitor else (
                    self._global_monitor * playback.per_sound_gain
                )

            if n > 0:
                chunk = data[voice.pos:voice.pos + n] * g
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
            with self._lock:
                if voice in playback.voices:
                    playback.voices.remove(voice)
                if not playback.voices:
                    self._playbacks.pop(playback.id, None)

        try:
            info = (
                sd.query_devices(device, "output") if device is not None
                else sd.query_devices(kind="output")
            )
            dev_ch = int(info.get("max_output_channels") or playback.channels)
            out_channels = max(1, min(playback.channels, dev_ch)) if playback.channels > 1 else min(2, dev_ch) or 1
        except Exception:
            out_channels = playback.channels

        try:
            stream = sd.OutputStream(
                samplerate=playback.sr,
                channels=out_channels,
                device=device,
                dtype="float32",
                callback=callback,
                finished_callback=finished,
            )
        except Exception:
            return False

        voice.stream = stream
        playback.voices.append(voice)
        try:
            stream.start()
            return True
        except Exception:
            if voice in playback.voices:
                playback.voices.remove(voice)
            try: stream.close()
            except Exception: pass
            return False

    # ─── Per-playback control ──────────────────────────────────────────────
    def pause(self, pb_id: str) -> bool:
        with self._lock:
            pb = self._playbacks.get(pb_id)
        if not pb:
            return False
        pb.paused = True
        return True

    def resume(self, pb_id: str) -> bool:
        with self._lock:
            pb = self._playbacks.get(pb_id)
        if not pb:
            return False
        pb.paused = False
        return True

    def stop(self, pb_id: str) -> bool:
        with self._lock:
            pb = self._playbacks.pop(pb_id, None)
        if not pb:
            return False
        self._silence_and_teardown([pb])
        return True

    def seek(self, pb_id: str, pos_sec: float) -> bool:
        with self._lock:
            pb = self._playbacks.get(pb_id)
        if not pb:
            return False
        target_sec = max(0.0, min(pb.duration_sec, float(pos_sec)))
        # Position in current_data = original_seconds / speed * sr
        new_pos = int(target_sec / pb.speed * pb.sr)
        new_pos = max(0, min(new_pos, len(pb.current_data)))
        for v in pb.voices:
            v.pos = new_pos
        return True

    def set_speed(self, pb_id: str, speed: float) -> bool:
        speed = max(0.25, min(4.0, float(speed)))
        with self._lock:
            pb = self._playbacks.get(pb_id)
        if not pb:
            return False
        if abs(pb.speed - speed) < 0.005:
            return True
        # Remember real-time position in the ORIGINAL timeline so we can land
        # on the same audio moment after re-stretching.
        cur_orig_sec = pb.position_sec
        new_data = _time_stretch(pb.original_data, pb.sr, speed)
        if new_data is pb.original_data and speed != 1.0:
            # audiotsm missing — bail rather than silently lying about speed.
            return False
        new_pos = int(cur_orig_sec / speed * pb.sr)
        new_pos = max(0, min(new_pos, len(new_data)))

        was_paused = pb.paused
        pb.paused = True  # block callbacks during the swap
        try:
            pb.current_data = new_data
            pb.speed = speed
            for v in pb.voices:
                v.pos = new_pos
        finally:
            pb.paused = was_paused
        return True

    # ─── Bulk control ──────────────────────────────────────────────────────
    def pause_all(self):
        with self._lock:
            for pb in self._playbacks.values():
                pb.paused = True

    def resume_all(self):
        with self._lock:
            for pb in self._playbacks.values():
                pb.paused = False

    def stop_all(self):
        with self._lock:
            pbs = list(self._playbacks.values())
            self._playbacks.clear()
        self._silence_and_teardown(pbs)

    def _silence_and_teardown(self, pbs: list[Playback]):
        """Two-phase stop: kill the audio output immediately, sweep streams later.

        sounddevice's OutputStream.stop()/close() block for a buffer-drain's
        worth of time — sequencing them per-voice is what made 'Stop tout'
        feel staggered. Flipping `paused=True` first means the callback
        emits zeros within one tick (~10 ms), so the user hears silence
        instantly. The actual stream teardown happens off-thread.
        """
        if not pbs:
            return
        streams = []
        for pb in pbs:
            pb.paused = True
            streams.extend(v.stream for v in pb.voices)
            pb.voices = []
        if streams:
            threading.Thread(
                target=self._teardown_streams, args=(streams,), daemon=True,
            ).start()

    @staticmethod
    def _teardown_streams(streams):
        for s in streams:
            try: s.stop()
            except Exception: pass
            try: s.close()
            except Exception: pass

    # ─── Decode (ffmpeg subprocess → soundfile) ────────────────────────────
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
        # old libsndfile). Direct ffmpeg → wav → soundfile, no ffprobe needed.
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
