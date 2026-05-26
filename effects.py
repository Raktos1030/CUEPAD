"""Effects chain — offline (apply_chain) + live streaming (LiveEffects).

The offline path is pure functions operating on a whole (samples, channels)
float32 buffer; effects run in a fixed order. The live path is a class with
persistent state — IIR filter zi, ring-mod / tremolo phase accumulators,
echo delay line, Schroeder-style reverb — so each callback chunk picks up
where the previous one left off without clicks.

Pitch shift is offline-only for now: streaming phase-vocoder PSOLA isn't
shipped yet, so LiveEffects skips it.
"""
from __future__ import annotations

import threading

import numpy as np


# ─── Public API ────────────────────────────────────────────────────────────

def apply_chain(data: np.ndarray, sr: int, cfg: dict | None) -> np.ndarray:
    """Run every configured effect on `data`. No-op if cfg is None/empty."""
    if not cfg:
        return data
    out = data
    # Pitch first (changes the timbre everything else operates on).
    if cfg.get("pitch_semitones"):
        out = pitch_shift(out, sr, float(cfg["pitch_semitones"]))
    # Tone shaping.
    if cfg.get("telephone"):
        out = bandpass(out, sr, 300.0, 3400.0)
    if cfg.get("lowpass_hz"):
        out = lowpass(out, sr, float(cfg["lowpass_hz"]))
    if cfg.get("highpass_hz"):
        out = highpass(out, sr, float(cfg["highpass_hz"]))
    # Modulation.
    if cfg.get("robot_hz"):
        out = ring_mod(out, sr, float(cfg["robot_hz"]))
    if cfg.get("tremolo_hz"):
        out = tremolo(out, sr, float(cfg["tremolo_hz"]),
                      float(cfg.get("tremolo_depth", 0.5)))
    # Saturation.
    if cfg.get("distortion"):
        out = distortion(out, float(cfg["distortion"]))
    # Time-domain space.
    if cfg.get("echo_ms"):
        out = echo(out, sr,
                   float(cfg["echo_ms"]),
                   float(cfg.get("echo_feedback", 0.4)),
                   float(cfg.get("echo_mix", 0.35)))
    if cfg.get("reverb"):
        out = reverb(out, sr, float(cfg["reverb"]))
    return out


# ─── Pitch / time ──────────────────────────────────────────────────────────

def pitch_shift(data: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by ±semitones, preserve clip length.

    Resample by 2^(s/12) then time-stretch back to the original length —
    cheap PSOLA-style trick that's plenty for soundboard fun.
    """
    if abs(semitones) < 0.05:
        return data
    factor = 2.0 ** (semitones / 12.0)

    n_orig = data.shape[0]
    n_ch = data.shape[1] if data.ndim > 1 else 1
    new_len = max(2, int(round(n_orig / factor)))

    src_idx = np.arange(n_orig, dtype=np.float64)
    tgt_idx = np.linspace(0.0, n_orig - 1, new_len)
    if data.ndim == 1:
        resampled = np.interp(tgt_idx, src_idx, data).astype(np.float32)
        resampled_2d = resampled.reshape(-1, 1)
    else:
        resampled = np.column_stack([
            np.interp(tgt_idx, src_idx, data[:, c]) for c in range(n_ch)
        ]).astype(np.float32)
        resampled_2d = resampled

    # Time-stretch back to original length while keeping the new pitch.
    try:
        import audiotsm
        from audiotsm.io.array import ArrayReader, ArrayWriter
    except Exception:
        return data
    reader = ArrayReader(resampled_2d.T.astype(np.float32))
    writer = ArrayWriter(n_ch)
    tsm = audiotsm.wsola(n_ch, speed=1.0 / factor)
    tsm.run(reader, writer)
    out = writer.data.T  # → (samples, channels)
    # WSOLA can over/undershoot by a few frames — clamp to original length so
    # downstream playback math doesn't have to special-case it.
    if out.shape[0] >= n_orig:
        out = out[:n_orig]
    else:
        pad = np.zeros((n_orig - out.shape[0], n_ch), dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    if data.ndim == 1:
        out = out[:, 0]
    return out.astype(np.float32)


# ─── Filters ───────────────────────────────────────────────────────────────

def _apply_sos(data: np.ndarray, sos) -> np.ndarray:
    from scipy import signal
    if data.ndim == 1:
        return signal.sosfilt(sos, data).astype(np.float32)
    out = np.zeros_like(data)
    for c in range(data.shape[1]):
        out[:, c] = signal.sosfilt(sos, data[:, c])
    return out


def lowpass(data: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    from scipy import signal
    nyq = sr * 0.5
    cutoff = max(20.0, min(nyq - 100.0, cutoff_hz))
    sos = signal.butter(4, cutoff / nyq, btype="low", output="sos")
    return _apply_sos(data, sos)


def highpass(data: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    from scipy import signal
    nyq = sr * 0.5
    cutoff = max(20.0, min(nyq - 100.0, cutoff_hz))
    sos = signal.butter(4, cutoff / nyq, btype="high", output="sos")
    return _apply_sos(data, sos)


def bandpass(data: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    from scipy import signal
    nyq = sr * 0.5
    low = max(20.0, min(nyq - 200.0, low_hz))
    high = max(low + 50.0, min(nyq - 100.0, high_hz))
    sos = signal.butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return _apply_sos(data, sos)


# ─── Modulation ────────────────────────────────────────────────────────────

def ring_mod(data: np.ndarray, sr: int, freq_hz: float) -> np.ndarray:
    """Multiply by a sine carrier — robotic / metallic tone."""
    n = data.shape[0]
    t = np.arange(n, dtype=np.float32) / sr
    carrier = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
    if data.ndim == 1:
        return (data * carrier).astype(np.float32)
    return (data * carrier[:, None]).astype(np.float32)


def tremolo(data: np.ndarray, sr: int, rate_hz: float, depth: float) -> np.ndarray:
    """Amplitude modulation at sub-audio rate."""
    depth = max(0.0, min(1.0, depth))
    n = data.shape[0]
    t = np.arange(n, dtype=np.float32) / sr
    env = (1.0 - depth) + depth * 0.5 * (1.0 + np.sin(2.0 * np.pi * rate_hz * t))
    env = env.astype(np.float32)
    if data.ndim == 1:
        return (data * env).astype(np.float32)
    return (data * env[:, None]).astype(np.float32)


# ─── Saturation ────────────────────────────────────────────────────────────

def distortion(data: np.ndarray, amount: float) -> np.ndarray:
    """tanh waveshaping. `amount` 0..1 maps to drive 1..10."""
    amount = max(0.0, min(1.0, amount))
    if amount < 0.01:
        return data
    drive = 1.0 + amount * 9.0
    # Compensate so peaks don't explode.
    makeup = 1.0 / (1.0 + amount * 0.8)
    return (np.tanh(data * drive) * makeup).astype(np.float32)


# ─── Time-domain space ─────────────────────────────────────────────────────

def echo(data: np.ndarray, sr: int, delay_ms: float,
         feedback: float, mix: float) -> np.ndarray:
    """Single-tap feedback delay. `feedback` ∈ [0,0.95], `mix` ∈ [0,1]."""
    delay = int(delay_ms * sr / 1000.0)
    if delay < 1 or delay >= data.shape[0]:
        return data
    feedback = max(0.0, min(0.95, feedback))
    mix = max(0.0, min(1.0, mix))
    out = data.astype(np.float32).copy()
    n = out.shape[0]
    # Each iteration adds another tap, attenuated by feedback. Bail when the
    # tap energy is inaudible (<-80 dB) instead of pre-computing tap count.
    gain = mix
    src = data.astype(np.float32)
    for tap in range(1, 32):
        gain *= feedback if tap > 1 else 1.0
        if gain < 1e-4:
            break
        start = delay * tap
        if start >= n:
            break
        if data.ndim == 1:
            out[start:] += src[:n - start] * gain
        else:
            out[start:, :] += src[:n - start, :] * gain
    return np.clip(out, -1.0, 1.0).astype(np.float32)


_REVERB_IR_CACHE: dict[int, np.ndarray] = {}


def _reverb_ir(sr: int) -> np.ndarray:
    """Lazy synthetic impulse response — decaying coloured noise, ~1.4s tail."""
    ir = _REVERB_IR_CACHE.get(sr)
    if ir is not None:
        return ir
    rng = np.random.default_rng(0xC0FFEE)
    ir_len = int(1.4 * sr)
    base = rng.standard_normal(ir_len).astype(np.float32)
    # Exponential decay shapes the noise into a believable room tail.
    decay = np.exp(-5.5 * np.arange(ir_len, dtype=np.float32) / ir_len)
    base *= decay
    # Low-pass it so the tail sounds dark instead of hissy.
    try:
        from scipy import signal
        sos = signal.butter(2, 6000.0 / (sr * 0.5), btype="low", output="sos")
        base = signal.sosfilt(sos, base).astype(np.float32)
    except Exception:
        pass
    peak = float(np.abs(base).max()) or 1.0
    base *= 0.35 / peak
    _REVERB_IR_CACHE[sr] = base
    return base


def reverb(data: np.ndarray, sr: int, mix: float) -> np.ndarray:
    """FFT-convolution reverb against a cached synthetic IR.

    Schroeder-style IIR reverbs are clean but every `lfilter` call costs
    O(N · D) with D ≈ samples-per-comb-delay, which made a single render of
    a 2 s stereo clip take ~750 ms on a modern CPU. fftconvolve runs in
    O(N log N) and lands the same render in ~30 ms.
    """
    mix = max(0.0, min(1.0, mix))
    if mix < 0.01:
        return data
    try:
        from scipy.signal import fftconvolve
    except Exception:
        return data
    ir = _reverb_ir(sr)
    if data.ndim == 1:
        wet = fftconvolve(data, ir, mode="full")[: data.shape[0]]
    else:
        wet = np.column_stack([
            fftconvolve(data[:, c], ir, mode="full")[: data.shape[0]]
            for c in range(data.shape[1])
        ])
    out = (1.0 - mix) * data + mix * wet
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# Live (streaming) path
# ───────────────────────────────────────────────────────────────────────────

class LiveEffects:
    """Streaming effects chain — process(chunk) keeps per-effect state between
    calls so filters don't click and oscillators don't reset on every block.

    Designed to be hot-swappable: the audio thread calls process(); the UI
    thread calls update_config() under the same lock.

    Pitch shift is intentionally absent — streaming phase-vocoder isn't
    shipped yet. Other effects mirror the offline implementations.
    """

    # Max echo we'll allocate room for, regardless of slider position.
    MAX_ECHO_SEC = 2.0
    REVERB_MAX_DELAY_SEC = 0.06  # widest Schroeder comb delay we use

    def __init__(self, sr: int, channels: int = 2):
        self.sr = int(sr)
        self.channels = int(channels)
        self._lock = threading.Lock()
        self._config: dict = {}

        # Filter zi state. Recomputed lazily when cutoffs change.
        self._lp_zi = None;  self._lp_sos = None;  self._lp_hz = None
        self._hp_zi = None;  self._hp_sos = None;  self._hp_hz = None
        self._bp_zi = None;  self._bp_sos = None;  self._bp_hz = None  # telephone preset

        # Oscillator phases — sub-sample float, advanced per chunk.
        self._ring_phase = 0.0
        self._trem_phase = 0.0

        # Echo ring buffer.
        echo_len = int(self.MAX_ECHO_SEC * self.sr) + 16
        self._echo_buf = np.zeros((echo_len, channels), dtype=np.float32)
        self._echo_pos = 0

        # Schroeder reverb — 4 parallel comb filters + 2 series allpass.
        # Each comb is a feedback delay line; persistent across chunks.
        comb_ms = [29.7, 37.1, 41.1, 43.7]
        comb_gain = [0.805, 0.795, 0.785, 0.775]
        self._comb_delays = [int(ms * 0.001 * self.sr) for ms in comb_ms]
        self._comb_gains  = comb_gain
        self._comb_bufs   = [np.zeros((d, channels), dtype=np.float32)
                             for d in self._comb_delays]
        self._comb_pos    = [0] * len(self._comb_delays)
        ap_specs = [(5.0, 0.7), (1.7, 0.7)]
        self._ap_delays = [int(ms * 0.001 * self.sr) for ms, _ in ap_specs]
        self._ap_gains  = [g for _, g in ap_specs]
        self._ap_bufs   = [np.zeros((d, channels), dtype=np.float32)
                           for d in self._ap_delays]
        self._ap_pos    = [0] * len(self._ap_delays)

    # ─── Public API ─────────────────────────────────────────────────────────
    def update_config(self, cfg: dict | None):
        with self._lock:
            new = dict(cfg or {})
            # Invalidate filter state if the cutoff changed — sosfilt_zi needs
            # to be recomputed for the new coefficients.
            if new.get("lowpass_hz") != self._lp_hz:
                self._lp_zi = self._lp_sos = None
                self._lp_hz = new.get("lowpass_hz")
            if new.get("highpass_hz") != self._hp_hz:
                self._hp_zi = self._hp_sos = None
                self._hp_hz = new.get("highpass_hz")
            if bool(new.get("telephone")) != (self._bp_hz is not None):
                self._bp_zi = self._bp_sos = None
                self._bp_hz = (300.0, 3400.0) if new.get("telephone") else None
            self._config = new

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process one (frames, channels) block. Returns same shape."""
        with self._lock:
            cfg = self._config
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        # Match channel count to our state — if mic is mono but our state is
        # stereo (or vice versa), broadcast.
        if chunk.shape[1] != self.channels:
            if chunk.shape[1] == 1 and self.channels == 2:
                chunk = np.repeat(chunk, 2, axis=1)
            elif chunk.shape[1] == 2 and self.channels == 1:
                chunk = chunk.mean(axis=1, keepdims=True)
        if not cfg:
            return chunk

        out = chunk
        if cfg.get("telephone"):
            out = self._bp_step(out)
        if cfg.get("lowpass_hz"):
            out = self._lp_step(out, float(cfg["lowpass_hz"]))
        if cfg.get("highpass_hz"):
            out = self._hp_step(out, float(cfg["highpass_hz"]))
        if cfg.get("robot_hz"):
            out = self._ring_step(out, float(cfg["robot_hz"]))
        if cfg.get("tremolo_hz"):
            out = self._trem_step(out, float(cfg["tremolo_hz"]),
                                  float(cfg.get("tremolo_depth", 0.5)))
        if cfg.get("distortion"):
            out = distortion(out, float(cfg["distortion"]))
        if cfg.get("echo_ms"):
            out = self._echo_step(out,
                                  float(cfg["echo_ms"]),
                                  float(cfg.get("echo_feedback", 0.4)),
                                  float(cfg.get("echo_mix", 0.35)))
        if cfg.get("reverb"):
            out = self._reverb_step(out, float(cfg["reverb"]))
        # Safety clip — long chains can push past 1.0 even with makeups.
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    # ─── Filters ────────────────────────────────────────────────────────────
    def _ensure_sos(self, kind: str, *args):
        from scipy import signal
        nyq = self.sr * 0.5
        if kind == "low":
            cut = max(20.0, min(nyq - 100.0, args[0]))
            sos = signal.butter(4, cut / nyq, btype="low", output="sos")
        elif kind == "high":
            cut = max(20.0, min(nyq - 100.0, args[0]))
            sos = signal.butter(4, cut / nyq, btype="high", output="sos")
        else:  # band
            lo, hi = args
            lo = max(20.0, min(nyq - 200.0, lo))
            hi = max(lo + 50.0, min(nyq - 100.0, hi))
            sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        zi = signal.sosfilt_zi(sos)
        return sos, zi

    def _lp_step(self, x, cutoff):
        from scipy import signal
        if self._lp_sos is None or self._lp_hz != cutoff:
            self._lp_sos, zi = self._ensure_sos("low", cutoff)
            # Tile zi for each channel — sosfilt_zi returns (n_sections, 2).
            self._lp_zi = np.repeat(zi[:, :, None], self.channels, axis=2)
            self._lp_hz = cutoff
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            out[:, c], self._lp_zi[:, :, c] = signal.sosfilt(
                self._lp_sos, x[:, c], zi=self._lp_zi[:, :, c])
        return out

    def _hp_step(self, x, cutoff):
        from scipy import signal
        if self._hp_sos is None or self._hp_hz != cutoff:
            self._hp_sos, zi = self._ensure_sos("high", cutoff)
            self._hp_zi = np.repeat(zi[:, :, None], self.channels, axis=2)
            self._hp_hz = cutoff
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            out[:, c], self._hp_zi[:, :, c] = signal.sosfilt(
                self._hp_sos, x[:, c], zi=self._hp_zi[:, :, c])
        return out

    def _bp_step(self, x):
        from scipy import signal
        if self._bp_sos is None:
            self._bp_sos, zi = self._ensure_sos("band", 300.0, 3400.0)
            self._bp_zi = np.repeat(zi[:, :, None], self.channels, axis=2)
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            out[:, c], self._bp_zi[:, :, c] = signal.sosfilt(
                self._bp_sos, x[:, c], zi=self._bp_zi[:, :, c])
        return out

    # ─── Modulation (phase-continuous across chunks) ───────────────────────
    def _ring_step(self, x, freq_hz):
        n = x.shape[0]
        t = (self._ring_phase + np.arange(n, dtype=np.float64)) / self.sr
        carrier = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
        self._ring_phase = (self._ring_phase + n) % (self.sr * 1e6)
        return (x * carrier[:, None]).astype(np.float32)

    def _trem_step(self, x, rate_hz, depth):
        depth = max(0.0, min(1.0, depth))
        n = x.shape[0]
        t = (self._trem_phase + np.arange(n, dtype=np.float64)) / self.sr
        env = ((1.0 - depth) + depth * 0.5 *
               (1.0 + np.sin(2.0 * np.pi * rate_hz * t))).astype(np.float32)
        self._trem_phase = (self._trem_phase + n) % (self.sr * 1e6)
        return (x * env[:, None]).astype(np.float32)

    # ─── Echo (ring buffer, vectorised when delay > block size) ────────────
    def _echo_step(self, x, delay_ms, feedback, mix):
        n = x.shape[0]
        L = self._echo_buf.shape[0]
        delay = max(1, min(int(delay_ms * self.sr / 1000.0), L - n - 1))
        feedback = max(0.0, min(0.95, feedback))
        mix = max(0.0, min(1.0, mix))
        buf = self._echo_buf
        pos = self._echo_pos

        # In all realistic settings delay >> block_size, so the read window
        # and the write window don't overlap and we can blit instead of
        # looping sample-by-sample.
        if delay >= n:
            delayed = _ring_read(buf, (pos - delay) % L, n)
            out = x + mix * delayed
            _ring_write(buf, pos, x + feedback * delayed)
        else:
            # Slow path for the unusual case of a tiny delay (echo flips into
            # a comb-y artifact zone here, but we keep the behaviour sane).
            out = np.empty_like(x)
            for i in range(n):
                w = (pos + i) % L
                r = (w - delay) % L
                d = buf[r]
                out[i] = x[i] + mix * d
                buf[w] = x[i] + feedback * d
        self._echo_pos = (pos + n) % L
        return out.astype(np.float32)

    # ─── Reverb (Schroeder, persistent state, vectorised) ──────────────────
    def _comb_filter_step(self, x, idx):
        delay = self._comb_delays[idx]
        gain  = self._comb_gains[idx]
        buf   = self._comb_bufs[idx]
        pos   = self._comb_pos[idx]
        n = x.shape[0]
        if n <= delay:
            delayed = _ring_read(buf, pos, n)
            new = x + gain * delayed
            _ring_write(buf, pos, new)
            out = new
        else:
            out = np.empty_like(x)
            for i in range(n):
                r = (pos + i) % delay
                d = buf[r]
                new = x[i] + gain * d
                buf[r] = new
                out[i] = new
        self._comb_pos[idx] = (pos + n) % delay
        return out

    def _allpass_filter_step(self, x, idx):
        # Canonical 1-delay Schroeder allpass:
        #   w[n] = x[n] + g * w[n-D]
        #   y[n] = -g * x[n] + w[n-D]
        delay = self._ap_delays[idx]
        gain  = self._ap_gains[idx]
        buf   = self._ap_bufs[idx]
        pos   = self._ap_pos[idx]
        n = x.shape[0]
        if n <= delay:
            w_old = _ring_read(buf, pos, n)
            w_new = x + gain * w_old
            _ring_write(buf, pos, w_new)
            out = -gain * x + w_old
        else:
            out = np.empty_like(x)
            for i in range(n):
                r = (pos + i) % delay
                w_old = buf[r]
                w_new = x[i] + gain * w_old
                buf[r] = w_new
                out[i] = -gain * x[i] + w_old
        self._ap_pos[idx] = (pos + n) % delay
        return out

    def _reverb_step(self, x, mix):
        mix = max(0.0, min(1.0, mix))
        if mix < 0.01:
            return x
        wet = np.zeros_like(x)
        for i in range(len(self._comb_delays)):
            wet += self._comb_filter_step(x, i)
        wet /= len(self._comb_delays)
        for i in range(len(self._ap_delays)):
            wet = self._allpass_filter_step(wet, i)
        return ((1.0 - mix) * x + mix * wet).astype(np.float32)


def _ring_read(buf: np.ndarray, start: int, n: int) -> np.ndarray:
    """Read n samples from a circular buffer starting at `start`, handling wrap."""
    L = buf.shape[0]
    start %= L
    end = start + n
    if end <= L:
        return buf[start:end].copy()
    head = L - start
    out = np.empty((n,) + buf.shape[1:], dtype=buf.dtype)
    out[:head] = buf[start:]
    out[head:] = buf[: n - head]
    return out


def _ring_write(buf: np.ndarray, start: int, data: np.ndarray) -> None:
    """Write data into a circular buffer at `start`, handling wrap."""
    L = buf.shape[0]
    n = data.shape[0]
    start %= L
    end = start + n
    if end <= L:
        buf[start:end] = data
    else:
        head = L - start
        buf[start:] = data[:head]
        buf[: n - head] = data[head:]
