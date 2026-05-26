"""Per-sound effects chain — applied at play time to the decoded audio buffer.

Every effect is a pure function `f(data, sr, *params) -> data` operating on a
float32 ndarray shaped (samples, channels). The chain runs in a fixed order
(pitch → tone shaping → modulation → time-domain → space) so chained presets
are predictable. `apply_chain(data, sr, cfg)` is the public entry point.

Heavy lifting (filtering, time-stretch) goes through scipy and audiotsm so
the per-play CPU cost is small even on long clips.
"""
from __future__ import annotations

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
