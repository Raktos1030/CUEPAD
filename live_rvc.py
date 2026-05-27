"""Live RVC — mic → voice changer → virtual cable, in real time(-ish).

The audio thread (sounddevice duplex callback) only touches ring buffers:
push 16 kHz mono input frames into the in-ring, pull tgt_sr frames out of
the out-ring. RVC inference is way too slow to run inside an audio
callback (even on GPU it's 50-200 ms per chunk), so a worker thread does
the heavy lifting:

    1. Wait until `chunk_samples + crossfade_samples` of mic audio is
       available in the in-ring (~ chunk_ms milliseconds of lookahead).
    2. Read them out (without consuming the crossfade tail yet — the
       next chunk needs it for context too).
    3. Run them through VoiceChanger.process_chunk(), which calls the
       vendored RVC pipeline.
    4. Crossfade the first `crossfade_samples` of the new output against
       the saved tail of the previous output; push the crossfaded prefix
       and the body of the new output into the out-ring.
    5. Save the tail of the new output for the next crossfade.

End-to-end latency = chunk_ms + crossfade_ms + inference time. On CPU
expect ~1.5-3 s (slow). On AMD GPU via torch-directml, ~600-900 ms.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np


CHUNK_PRESETS = {
    # ms of audio per inference pass → trades latency for context
    "low":    400,   # punchy but more chunk overhead
    "medium": 700,
    "high":   1100,  # smoother / more natural prosody
    "safe":   1500,  # last resort if low/medium glitches under load
}
CROSSFADE_MS = 80    # equal-power crossfade between successive chunks


class _RingBuffer:
    """Thread-safe float32 ring buffer for streaming audio frames."""

    def __init__(self, capacity_samples: int, channels: int = 1):
        self._cap = int(capacity_samples)
        self._ch  = int(channels)
        self._buf = np.zeros((self._cap, self._ch), dtype=np.float32)
        self._w = 0
        self._r = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = data.shape[0]
        with self._lock:
            free = self._cap - self._filled
            if n > free:
                # Drop oldest samples (preferable to dropping new ones for
                # mic input — we want the freshest frames available).
                drop = n - free
                self._r = (self._r + drop) % self._cap
                self._filled -= drop
            end = self._w + n
            if end <= self._cap:
                self._buf[self._w:end] = data
            else:
                head = self._cap - self._w
                self._buf[self._w:] = data[:head]
                self._buf[:n - head] = data[head:]
            self._w = (self._w + n) % self._cap
            self._filled += n
            return n

    def read(self, n: int) -> np.ndarray | None:
        with self._lock:
            if self._filled < n:
                return None
            out = np.empty((n, self._ch), dtype=np.float32)
            end = self._r + n
            if end <= self._cap:
                out[:] = self._buf[self._r:end]
            else:
                head = self._cap - self._r
                out[:head] = self._buf[self._r:]
                out[head:] = self._buf[:n - head]
            self._r = (self._r + n) % self._cap
            self._filled -= n
            return out

    def peek(self, n: int) -> np.ndarray | None:
        with self._lock:
            if self._filled < n:
                return None
            out = np.empty((n, self._ch), dtype=np.float32)
            end = self._r + n
            if end <= self._cap:
                out[:] = self._buf[self._r:end]
            else:
                head = self._cap - self._r
                out[:head] = self._buf[self._r:]
                out[head:] = self._buf[:n - head]
            return out

    def consume(self, n: int):
        with self._lock:
            n = min(n, self._filled)
            self._r = (self._r + n) % self._cap
            self._filled -= n

    @property
    def available(self) -> int:
        return self._filled

    def reset(self):
        with self._lock:
            self._w = self._r = self._filled = 0
            self._buf.fill(0)


def _equal_power_crossfade(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_in, fade_out) windows of length n that sum to constant
    power — pleasant for voice transitions."""
    t = np.linspace(0.0, np.pi * 0.5, n, dtype=np.float32)
    return np.sin(t), np.cos(t)


class LiveRvcEngine:
    """One duplex stream + one inference worker. Single-voice at a time."""

    SR_IN  = 16000   # what HuBERT wants
    SR_DUP = 48000   # what the audio stream runs at on both sides

    def __init__(self, voice_changer):
        self.vc = voice_changer
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._in_ring:  Optional[_RingBuffer] = None
        self._out_ring: Optional[_RingBuffer] = None
        self._resampler_in  = None
        self._resampler_out = None

        # Inference config (hot-swappable via update_params)
        self._params = {
            "f0_up_key":     0,
            "f0_method":     "rmvpe",
            "index_rate":    0.5,
            "protect":       0.33,
            "rms_mix_rate":  0.25,
            "filter_radius": 3,
        }
        self._voice: Optional[str] = None
        self._chunk_ms = CHUNK_PRESETS["medium"]
        self._cf_ms = CROSSFADE_MS

        # Tail saved between chunks for crossfade.
        self._prev_tail_out: Optional[np.ndarray] = None

        # Telemetry
        self._underruns = 0
        self._chunk_count = 0
        self._avg_infer_ms = 0.0
        self._started_ts: Optional[float] = None
        self._last_error: Optional[str] = None

    # ─── Status / params ────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "running":       self._stream is not None,
            "voice":         self._voice,
            "chunk_ms":      self._chunk_ms,
            "crossfade_ms":  self._cf_ms,
            "params":        dict(self._params),
            "underruns":     self._underruns,
            "chunks":        self._chunk_count,
            "avg_infer_ms":  round(self._avg_infer_ms, 1),
            "uptime_sec":    round(time.monotonic() - self._started_ts, 1) if self._started_ts else 0.0,
            "error":         self._last_error,
            "tgt_sr":        self.vc.streaming_target_sr(),
            "presets":       list(CHUNK_PRESETS.keys()),
            "last_peak":     round(float(getattr(self.vc, "last_chunk_peak", 0.0)), 4),
            # ONNX/DirectML inference time breakdown (None on torch path).
            "onnx_timings":  getattr(self.vc, "last_onnx_timings", None),
            # Which compute path is actually live ('onnx' vs 'torch').
            "backend":       "onnx" if getattr(self.vc, "_onnx_session", None) else "torch",
            "device_label":  getattr(self.vc, "_device_label", "?"),
        }

    def update_params(self, **kw):
        with self._lock:
            for k, v in kw.items():
                if k in self._params:
                    self._params[k] = v

    # ─── Lifecycle ──────────────────────────────────────────────────────────
    def start(self, input_dev, output_dev, voice_name: str,
              latency: str = "medium", **params) -> tuple[bool, Optional[str]]:
        try:
            import sounddevice as sd
        except Exception as e:
            return False, f"sounddevice indisponible: {e}"

        if self._stream is not None:
            self.stop()

        # 1. Warm up the voice model + HuBERT before opening the audio stream
        #    — otherwise the first chunk's inference would underflow the
        #    output ring instantly.
        ok, err = self.vc.prepare_for_streaming(voice_name)
        if not ok:
            self._last_error = err
            return False, err
        tgt_sr = self.vc.streaming_target_sr()
        if not tgt_sr:
            return False, "Voix non chargée (target SR inconnu)"

        self._voice = voice_name
        self._chunk_ms = CHUNK_PRESETS.get(latency, CHUNK_PRESETS["medium"])
        self.update_params(**{k: v for k, v in params.items() if k in self._params})

        # Pick a duplex sample rate the devices both support.
        try:
            in_info  = sd.query_devices(input_dev,  "input")
            out_info = sd.query_devices(output_dev, "output")
            for cand in (48000, 44100, int(in_info.get("default_samplerate") or 48000)):
                try:
                    sd.check_input_settings(device=input_dev, samplerate=cand, channels=1)
                    sd.check_output_settings(device=output_dev, samplerate=cand, channels=1)
                    duplex_sr = cand
                    break
                except Exception:
                    continue
            else:
                duplex_sr = 48000
        except Exception as e:
            return False, f"Périphérique introuvable: {e}"

        # Ring buffers — keep ~5 s of audio to absorb worst-case stalls.
        self._in_ring  = _RingBuffer(self.SR_IN * 5, channels=1)
        self._out_ring = _RingBuffer(tgt_sr * 5,     channels=1)
        self._prev_tail_out = None
        self._underruns = 0
        self._chunk_count = 0
        self._avg_infer_ms = 0.0
        self._last_error = None

        # 2. Audio callback: resample mic → 16 kHz → in_ring,
        #    out_ring → tgt_sr → resample to duplex_sr → outdata.
        from scipy.signal import resample_poly
        in_resamp_num, in_resamp_den = self.SR_IN, duplex_sr
        out_resamp_num, out_resamp_den = duplex_sr, tgt_sr
        engine = self

        def callback(indata, outdata, frames, time_info, status):
            if status:
                engine._underruns += 1
            try:
                mono_in = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
                if duplex_sr != engine.SR_IN:
                    mono_in = resample_poly(mono_in, in_resamp_num, in_resamp_den).astype(np.float32)
                engine._in_ring.write(mono_in)

                # Consume exactly the number of tgt_sr samples needed to
                # produce `frames` duplex samples. The previous code added
                # a +4 slack which leaked ~5 tgt_sr samples per callback
                # into discard land, draining the out-ring faster than the
                # worker could fill it (audible underruns after ~1 minute).
                if duplex_sr == tgt_sr:
                    want = frames
                else:
                    want = int(np.ceil(frames * tgt_sr / duplex_sr))
                out_block = engine._out_ring.read(want)
                if out_block is None:
                    outdata.fill(0)
                    return
                mono_out = out_block[:, 0]
                if duplex_sr != tgt_sr:
                    mono_out = resample_poly(mono_out, out_resamp_num, out_resamp_den).astype(np.float32)
                # Trim/pad to exact frame count
                if mono_out.shape[0] < frames:
                    pad = np.zeros(frames - mono_out.shape[0], dtype=np.float32)
                    mono_out = np.concatenate([mono_out, pad])
                outdata[:] = mono_out[:frames, None]
            except Exception as e:
                engine._last_error = f"callback: {e}"
                outdata.fill(0)

        try:
            stream = sd.Stream(
                samplerate=duplex_sr,
                blocksize=1024,
                device=(input_dev, output_dev),
                channels=(1, 1),
                dtype="float32",
                callback=callback,
                latency="low",
            )
            stream.start()
        except Exception as e:
            return False, f"Démarrage stream: {e}"

        self._stream = stream
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, args=(tgt_sr,), daemon=True)
        self._worker.start()
        self._started_ts = time.monotonic()
        return True, None

    def stop(self):
        self._stop.set()
        w = self._worker; self._worker = None
        if w is not None:
            w.join(timeout=2.0)
        s = self._stream; self._stream = None
        self._started_ts = None
        if s is not None:
            try: s.stop()
            except Exception: pass
            try: s.close()
            except Exception: pass

    # ─── Worker thread ──────────────────────────────────────────────────────
    def _worker_loop(self, tgt_sr: int):
        chunk_n = int(self.SR_IN * self._chunk_ms / 1000.0)
        cf_n_in  = int(self.SR_IN * self._cf_ms / 1000.0)
        cf_n_out = int(tgt_sr     * self._cf_ms / 1000.0)
        # The worker needs (chunk + crossfade) samples available before it
        # can run a pass — the tail provides context for the NEXT chunk.
        needed = chunk_n + cf_n_in

        while not self._stop.is_set():
            if self._in_ring is None or self._out_ring is None:
                time.sleep(0.05); continue
            if self._in_ring.available < needed:
                time.sleep(0.01); continue

            block = self._in_ring.read(chunk_n)
            if block is None:
                time.sleep(0.01); continue
            # Read (peek-style) the crossfade tail — it stays in the buffer
            # so it doubles as the next chunk's beginning context.
            tail = self._in_ring.peek(cf_n_in)
            mono = block[:, 0]
            if tail is not None:
                mono = np.concatenate([mono, tail[:, 0]])

            with self._lock:
                p = dict(self._params)

            t0 = time.monotonic()
            try:
                out = self.vc.process_chunk(mono, **p)
            except Exception as e:
                self._last_error = f"infer: {e}"
                out = None
            else:
                # Mirror process_chunk's most recent diagnostic onto the
                # engine status — including None when the latest chunk was
                # fine, so an old error message doesn't stick on the UI.
                self._last_error = getattr(self.vc, "last_chunk_error", None)
            dt_ms = (time.monotonic() - t0) * 1000.0
            self._chunk_count += 1
            self._avg_infer_ms = (
                (self._avg_infer_ms * (self._chunk_count - 1) + dt_ms) / self._chunk_count
            )

            if out is None or out.size == 0:
                continue

            # Cross-fade prefix with the previous tail, write body to out_ring,
            # save new tail for next pass.
            cf = min(cf_n_out, out.shape[0] // 4)
            if self._prev_tail_out is not None and cf > 0:
                f_in, f_out = _equal_power_crossfade(cf)
                head_n = min(cf, self._prev_tail_out.shape[0])
                mixed = (
                    out[:head_n] * f_in[:head_n] +
                    self._prev_tail_out[-head_n:] * f_out[-head_n:]
                )
                body = out[head_n:-cf] if out.shape[0] > head_n + cf else out[head_n:head_n]
                self._out_ring.write(np.concatenate([mixed, body]))
            else:
                self._out_ring.write(out[:-cf] if cf > 0 else out)
            self._prev_tail_out = out[-cf:] if cf > 0 else None
