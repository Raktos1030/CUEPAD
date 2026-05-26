"""Live mic effects engine — duplex sounddevice stream piping input through
a LiveEffects chain to a configurable output device.

The audio thread reads `indata`, runs the cached LiveEffects (which holds its
own per-effect state across calls), and writes the result to `outdata`. The
UI thread mutates the effects config via update_config(); LiveEffects.process
takes a lock so we never read half-written config.

Block size is exposed as a "latency" setting (in milliseconds), trading
latency for stability. We rebuild the stream when the user changes the I/O
devices or the block size; we don't rebuild for effect changes.
"""
from __future__ import annotations

import threading
import time

from effects import LiveEffects


# Block sizes are picked so blk / 48000 hits the latency hint cleanly.
LATENCY_PRESETS = {
    "low":    256,   # ~5 ms per direction
    "medium": 512,   # ~11 ms
    "high":   1024,  # ~21 ms
    "safe":   2048,  # ~43 ms — for fragile machines
}


class LiveMicEngine:
    def __init__(self):
        self._stream = None
        self._fx: LiveEffects | None = None
        self._cfg: dict = {}
        self._input_dev = None
        self._output_dev = None
        self._sr = 48000
        self._blocksize = LATENCY_PRESETS["low"]
        self._lock = threading.Lock()
        self._last_error: str | None = None
        self._underruns = 0
        self._started_ts: float | None = None

    def status(self) -> dict:
        return {
            "running":     self._stream is not None,
            "input":       self._input_dev,
            "output":      self._output_dev,
            "samplerate":  self._sr,
            "blocksize":   self._blocksize,
            "latency_ms":  round(self._blocksize / self._sr * 1000.0, 1),
            "underruns":   self._underruns,
            "uptime_sec":  round(time.monotonic() - self._started_ts, 1) if self._started_ts else 0.0,
            "config":      dict(self._cfg),
            "error":       self._last_error,
        }

    def update_config(self, cfg: dict | None):
        with self._lock:
            self._cfg = dict(cfg or {})
            if self._fx is not None:
                self._fx.update_config(self._cfg)

    def start(self, input_dev, output_dev, latency: str = "low") -> tuple[bool, str | None]:
        try:
            import sounddevice as sd
        except Exception as e:
            return False, f"sounddevice indisponible: {e}"

        if self._stream is not None:
            self.stop()

        blocksize = LATENCY_PRESETS.get(latency, LATENCY_PRESETS["low"])

        # Resolve the device's preferred sample rate so we don't fight it.
        try:
            in_info  = sd.query_devices(input_dev,  "input")
            out_info = sd.query_devices(output_dev, "output")
            sr_candidates = [
                int(in_info.get("default_samplerate")  or 0),
                int(out_info.get("default_samplerate") or 0),
            ]
            sr = next((s for s in sr_candidates if s in (44100, 48000)), 48000)
        except Exception as e:
            self._last_error = f"Périph introuvable: {e}"
            return False, self._last_error

        # Mic comes in mono on most realistic setups; we lift to stereo for
        # the output so DAWs / virtual cables receive a normal signal.
        in_ch  = min(2, int(in_info.get("max_input_channels", 1))) or 1
        out_ch = min(2, int(out_info.get("max_output_channels", 2))) or 2

        fx = LiveEffects(sr, channels=out_ch)
        with self._lock:
            fx.update_config(self._cfg)
            self._fx = fx

        engine = self  # closure capture

        def callback(indata, outdata, frames, time_info, status):
            if status:
                # Drop the silent samples on the floor but record so we can
                # tell the UI we're under-running.
                engine._underruns += 1
            try:
                with engine._lock:
                    fx_ref = engine._fx
                # Lift mic input to the output channel count before processing.
                if indata.shape[1] == 1 and out_ch == 2:
                    block = indata.repeat(2, axis=1)
                elif indata.shape[1] == 2 and out_ch == 1:
                    block = indata.mean(axis=1, keepdims=True)
                else:
                    block = indata.copy()
                out = fx_ref.process(block) if fx_ref is not None else block
                outdata[:] = out[:frames]
            except Exception as e:
                engine._last_error = f"callback: {e}"
                outdata.fill(0)

        try:
            stream = sd.Stream(
                samplerate=sr,
                blocksize=blocksize,
                device=(input_dev, output_dev),
                channels=(in_ch, out_ch),
                dtype="float32",
                callback=callback,
                latency="low",
            )
            stream.start()
        except Exception as e:
            self._last_error = f"Démarrage stream: {e}"
            with self._lock:
                self._fx = None
            return False, self._last_error

        self._stream = stream
        self._input_dev = input_dev
        self._output_dev = output_dev
        self._sr = sr
        self._blocksize = blocksize
        self._underruns = 0
        self._started_ts = time.monotonic()
        self._last_error = None
        return True, None

    def stop(self):
        s = self._stream
        self._stream = None
        self._started_ts = None
        with self._lock:
            self._fx = None
        if s is not None:
            # Off-thread teardown for the same reason as AudioEngine.stop_all —
            # close() on sounddevice blocks long enough to be felt at the UI.
            threading.Thread(target=_close, args=(s,), daemon=True).start()


def _close(stream):
    try: stream.stop()
    except Exception: pass
    try: stream.close()
    except Exception: pass
