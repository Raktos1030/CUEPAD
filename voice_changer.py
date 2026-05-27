"""RVC offline voice changer.

Loads a user-supplied RVC model (.pth + .index) and converts the timbre of an
audio file to that voice. Inference is delegated to `rvc_python`, which
bundles the HuBERT feature extractor, RMVPE/Crepe/Harvest/PM pitch
extractors, and the RVC synthesizer. On first use it downloads the base
HuBERT / RMVPE assets (~540 MB) into its own data dir.

Models live as one subdirectory per voice:
    <voices_dir>/mbappe/mbappe.pth
    <voices_dir>/mbappe/added_*.index   (optional)

`import_voice(...)` ingests a flat .pth (+optional .index) the user drops via
the UI by moving them into a new subdir. `list_voices()` reports what's
ready. `convert_file(...)` runs inference off-thread (callers should already
be inside a worker — we don't spawn here so the call blocks).
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional


# Pitch extractor presets exposed to the UI — name → method passed to rvc.
PITCH_METHODS = ["rmvpe", "crepe", "harvest", "pm"]


class VoiceChanger:
    def __init__(self, voices_dir: Path):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._rvc = None
        self._current_model: Optional[str] = None
        self._lock = threading.Lock()
        self._init_error: Optional[str] = None
        self._init_ts: Optional[float] = None

    # ─── Status / introspection ────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "ready":        self._rvc is not None,
            "init_error":   self._init_error,
            "current":      self._current_model,
            "voices_dir":   str(self.voices_dir),
            "voice_count":  len(self.list_voices()),
            "pitch_methods": PITCH_METHODS,
        }

    def list_voices(self) -> list[dict]:
        """Each voice subdirectory containing at least one .pth is a voice."""
        out = []
        for sub in sorted(self.voices_dir.iterdir() if self.voices_dir.exists() else []):
            if not sub.is_dir() or sub.name.startswith("_") or sub.name.startswith("."):
                continue
            pths = list(sub.glob("*.pth"))
            if not pths:
                continue
            pth = pths[0]
            indexes = list(sub.glob("*.index"))
            out.append({
                "name":      sub.name,
                "pth":       pth.name,
                "size_mb":   round(pth.stat().st_size / 1024 / 1024, 1),
                "index":     indexes[0].name if indexes else None,
                "has_index": bool(indexes),
            })
        return out

    # ─── Model lifecycle ───────────────────────────────────────────────────
    def _ensure_engine(self):
        """Lazy-load rvc-python so the app starts fast even without RVC deps."""
        if self._rvc is not None:
            return
        with self._lock:
            if self._rvc is not None:
                return
            try:
                from rvc_python.infer import RVCInference
            except Exception as e:
                self._init_error = (
                    f"rvc-python introuvable ({e}). Installe-le avec : "
                    "pip install rvc-python (et torch si pas déjà fait)."
                )
                raise RuntimeError(self._init_error) from e
            try:
                t0 = time.monotonic()
                # The constructor auto-downloads HuBERT/RMVPE the first time
                # — this can take a few minutes on a slow connection.
                self._rvc = RVCInference(
                    models_dir=str(self.voices_dir),
                    device="cpu:0",
                )
                self._init_ts = time.monotonic() - t0
                self._init_error = None
            except Exception as e:
                self._init_error = f"Initialisation RVC échouée: {e}"
                raise

    def _load(self, voice_name: str):
        self._ensure_engine()
        if self._current_model == voice_name:
            return
        # set_models_dir to pick up any voices added since last call
        self._rvc.set_models_dir(str(self.voices_dir))
        self._rvc.load_model(voice_name)
        self._current_model = voice_name

    # ─── Voice file management ─────────────────────────────────────────────
    def import_voice(self, name: str, pth_src: Path,
                     index_src: Path | None = None) -> tuple[bool, Optional[str]]:
        """Copy a .pth (+ optional .index) into a new voice subdir."""
        safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        if not safe:
            return False, "Nom invalide"
        dest = self.voices_dir / safe
        if dest.exists() and any(dest.iterdir()):
            return False, f"Voix '{safe}' existe déjà"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pth_src, dest / Path(pth_src).name)
            if index_src is not None and Path(index_src).exists():
                shutil.copy2(index_src, dest / Path(index_src).name)
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            return False, str(e)
        return True, None

    def delete_voice(self, name: str) -> bool:
        target = self.voices_dir / name
        if not target.is_dir():
            return False
        # Don't blow away anything outside the voices dir even if `name` had ..
        if self.voices_dir.resolve() not in target.resolve().parents:
            return False
        if self._current_model == name:
            self._current_model = None
        shutil.rmtree(target, ignore_errors=True)
        return True

    # ─── Inference ─────────────────────────────────────────────────────────
    def convert_file(
        self,
        voice_name: str,
        input_path: str | Path,
        output_path: str | Path,
        *,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.5,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        filter_radius: int = 3,
    ) -> tuple[bool, Optional[str]]:
        """Convert `input_path` to `voice_name`, write WAV to `output_path`.
        Blocks for the duration of inference — caller is expected to be in a
        worker thread or background job."""
        with self._lock:
            try:
                self._load(voice_name)
                self._rvc.set_params(
                    f0up_key=int(f0_up_key),
                    f0method=str(f0_method),
                    index_rate=float(index_rate),
                    protect=float(protect),
                    rms_mix_rate=float(rms_mix_rate),
                    filter_radius=int(filter_radius),
                )
                self._rvc.infer_file(str(input_path), str(output_path))
                return True, None
            except Exception as e:
                return False, str(e)
