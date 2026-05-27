"""RVC offline voice changer — fairseq-free path.

The first version of this module wrapped `rvc-python`. That dragged in
fairseq, omegaconf 2.0.6 (broken metadata), hydra-core 1.0.x, antlr4
4.8, and a C++ build of fairseq's libbleu — Windows + Python 3.11 hit
"Microsoft Visual C++ Build Tools required" the moment pip tried to
compile fairseq. Nope.

This rewrite vendors the RVC model architectures (rvc_lib/) and loads
HuBERT via `transformers.HubertModel` using the `lengyue233/content-vec-best`
weights, which are the same ContentVec weights upstream RVC ships in the
fairseq format. Net result: no fairseq, no omegaconf, no hydra, no C++
compilation, install through plain pip wheels on Python 3.11 Windows.

Voices live as one subdirectory per voice under <voices_dir>:
    <voices_dir>/mbappe/mbappe.pth
    <voices_dir>/mbappe/added_*.index   (optional)

Base assets (HuBERT auto-downloaded by transformers, plus rmvpe.pt that
we download manually) cache to <voices_dir>/_base/.

Limitations:
- Only RVC v2 .pth models are supported (the common 2025 case). v1
  models need a `final_proj` 768→256 layer that ships inside fairseq's
  HuBERT checkpoint; the HuggingFace ContentVec weights don't include
  it. v1 voices will be refused with a clear error message.
- CPU inference only for now. AMD GPUs via DirectML would need
  torch-directml and a few more lines — left as a follow-up.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional


PITCH_METHODS = ["rmvpe", "crepe", "harvest", "pm"]

RMVPE_URL    = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
RMVPE_SIZE   = 181 * 1024 * 1024  # ~181 MB, used for the progress UI


def _list_directml_adapters() -> list[dict]:
    """Enumerate DirectML adapters by index. Windows can expose several:
    an integrated iGPU on Ryzen Radeon, a discrete card, and even
    'Microsoft Basic Render Driver'. Picking the wrong one (iGPU) is a
    common reason DirectML feels slower than CPU.
    """
    try:
        import torch_directml  # type: ignore
    except Exception:
        return []
    out = []
    try:
        n = torch_directml.device_count() if hasattr(torch_directml, "device_count") else 0
    except Exception:
        n = 0
    for i in range(n):
        try:
            name = (torch_directml.device_name(i)
                    if hasattr(torch_directml, "device_name") else f"adapter {i}")
        except Exception:
            name = f"adapter {i}"
        out.append({"index": i, "name": name})
    return out


def _detect_device(preferred: str = "auto") -> tuple[str, str]:
    """Pick the best PyTorch device for RVC inference.

    Returns (device_string, label) where:
        device_string is something Pipeline can pass to `.to(...)`,
        label is a short human-readable tag for the UI / logs.

    `preferred` accepts:
        'auto'           — try CUDA → DirectML[0] → CPU
        'cpu'
        'cuda'           — fail to CPU if no CUDA
        'dml' or 'dml:0' — DirectML adapter 0
        'dml:1', 'dml:N' — explicit DirectML adapter index
    """
    def _dml(idx: int) -> tuple[str, str] | None:
        try:
            import torch_directml  # type: ignore
            if hasattr(torch_directml, "is_available") and not torch_directml.is_available():
                return None
            count = torch_directml.device_count() if hasattr(torch_directml, "device_count") else 1
            if idx >= count:
                return None
            name = (torch_directml.device_name(idx)
                    if hasattr(torch_directml, "device_name") else f"adapter {idx}")
            return f"privateuseone:{idx}", f"GPU (DirectML: {name})"
        except Exception:
            return None

    if preferred and preferred != "auto":
        try:
            import torch
        except Exception:
            return "cpu", "CPU"
        if preferred == "cpu":
            return "cpu", "CPU"
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda:0", f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
        if preferred.startswith("dml"):
            idx = 0
            if ":" in preferred:
                try: idx = int(preferred.split(":", 1)[1])
                except ValueError: idx = 0
            r = _dml(idx)
            if r is not None:
                return r
        return "cpu", "CPU"

    # auto: CUDA → first DirectML adapter that isn't Microsoft's fallback → CPU
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0", f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    for ad in _list_directml_adapters():
        if "basic render" in ad["name"].lower():
            continue  # software renderer; never want this
        r = _dml(ad["index"])
        if r is not None:
            return r
    return "cpu", "CPU"


class _Config:
    """Minimal stand-in for rvc-python's Config object — Pipeline only reads
    a handful of attributes off this, so we hand-roll just those."""
    def __init__(self, device: str = "cpu", is_half: bool = False):
        self.device  = device
        self.is_half = is_half
        # Chunk sizes — for GPU we can afford larger windows (more parallelism
        # = better throughput); CPU stays conservative to keep RAM bounded.
        is_gpu = device != "cpu"
        self.x_pad    = 3  if is_gpu else 1
        self.x_query  = 10 if is_gpu else 6
        self.x_center = 60 if is_gpu else 38
        self.x_max    = 65 if is_gpu else 41


class VoiceChanger:
    def __init__(self, voices_dir: Path, device_pref: str = "auto"):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.voices_dir / "_base"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir = self.base_dir / "hf"
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        self._hubert     = None  # HubertContentExtractor
        self._net_g      = None  # vendored synthesizer
        self._pipeline   = None  # vendored Pipeline
        self._tgt_sr     = None
        self._if_f0      = 1
        self._version    = "v2"
        self._current    = None
        self._cpt        = None
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()
        self._threads_tuned = False

        self._device_pref = device_pref
        self._device, self._device_label = _detect_device(device_pref)

    def set_device_pref(self, pref: str):
        """Change preferred device. Forces a reload of the loaded voice next
        time `_load_voice()` is called. Holds self._lock so a worker thread
        mid-inference can't see a half-zeroed state (net_g None while the
        old pipeline still hands out tensors)."""
        if pref == self._device_pref:
            return
        with self._lock:
            self._device_pref = pref
            self._device, self._device_label = _detect_device(pref)
            self._unload_current()

    # ─── Status / introspection ────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "ready":        self._pipeline is not None,
            "init_error":   self._init_error,
            "current":      self._current,
            "voices_dir":   str(self.voices_dir),
            "voice_count":  len(self.list_voices()),
            "pitch_methods": PITCH_METHODS,
            "device":       self._device,
            "device_label": self._device_label,
            "device_pref":  self._device_pref,
            "dml_adapters": _list_directml_adapters(),
        }

    def list_voices(self) -> list[dict]:
        if not self.voices_dir.exists():
            return []
        out = []
        for sub in sorted(self.voices_dir.iterdir()):
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

    # ─── Voice file management ─────────────────────────────────────────────
    def import_voice(self, name: str, pth_src, index_src=None) -> tuple[bool, Optional[str]]:
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
        try:
            target.resolve().relative_to(self.voices_dir.resolve())
        except ValueError:
            return False  # path traversal guard
        if self._current == name:
            self._unload_current()
        shutil.rmtree(target, ignore_errors=True)
        return True

    # ─── Base asset download ───────────────────────────────────────────────
    def _ensure_rmvpe(self):
        """Download rmvpe.pt into the base dir if it's not there yet."""
        dest = self.base_dir / "rmvpe.pt"
        if dest.exists() and dest.stat().st_size > 100_000_000:
            return dest
        try:
            import requests
        except Exception as e:
            raise RuntimeError(
                "`requests` introuvable — installe les deps RVC : "
                "pip install -r requirements-rvc.txt"
            ) from e
        tmp = dest.with_suffix(".pt.part")
        try:
            with requests.get(RMVPE_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        f.write(chunk)
            tmp.replace(dest)
        except Exception as e:
            try: tmp.unlink()
            except Exception: pass
            raise RuntimeError(f"Téléchargement rmvpe.pt échoué: {e}") from e
        return dest

    # ─── Engine / model lifecycle ──────────────────────────────────────────
    def _unload_current(self):
        # Drop HuBERT alongside the synthesizer — otherwise a device change
        # leaves it on the old backend and the next inference crashes with
        # 'weight type (privateuseoneFloatType)' vs 'Input type (FloatTensor)'.
        self._net_g = None
        self._pipeline = None
        self._current = None
        self._cpt = None
        self._hubert = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_voice(self, voice_name: str):
        """Load the user .pth + bring up the matching synthesizer."""
        # Early-return only if the SAME voice is already loaded on the SAME
        # device — otherwise we'd skip a needed reload after a device change.
        if (
            self._current == voice_name
            and self._net_g is not None
            and self._pipeline is not None
            and getattr(self._pipeline, "device", None) == self._device
        ):
            return

        info = next((v for v in self.list_voices() if v["name"] == voice_name), None)
        if info is None:
            raise RuntimeError(f"Voix introuvable: {voice_name}")
        pth_path = self.voices_dir / voice_name / info["pth"]

        try:
            import torch
            import os
        except Exception as e:
            raise RuntimeError(
                "PyTorch n'est pas installé — lance : "
                "pip install -r requirements-rvc.txt"
            ) from e
        # PyTorch CPU defaults to 1 thread on Windows, which leaves a Ryzen
        # 9700X mostly idle during RVC inference. Bump to all physical cores
        # every load — set_num_threads is safe to re-set, so users toggling
        # GPU→CPU still get the boost even when the first load happened on
        # DirectML. set_num_interop_threads can only be set BEFORE any tensor
        # op has run; guard it once and don't worry if it fails later.
        try:
            n = os.cpu_count() or 8
            torch.set_num_threads(max(1, n // 2))
        except Exception:
            pass
        if not self._threads_tuned:
            try:
                torch.set_num_interop_threads(max(1, (os.cpu_count() or 8) // 4))
            except Exception:
                pass
            self._threads_tuned = True
        try:
            from rvc_lib.models import (
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            )
        except Exception as e:
            raise RuntimeError(f"rvc_lib import failed: {e}") from e

        cpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
        version = cpt.get("version", "v1")
        if version != "v2":
            raise RuntimeError(
                f"Voix '{voice_name}' est un modèle v{version[1:]}. Seuls les "
                "modèles RVC v2 sont supportés ici — le backend fairseq-free "
                "n'a pas le projecteur 'final_proj' des v1. Réentraîne ou "
                "convertis le modèle en v2."
            )
        if_f0 = cpt.get("f0", 1)
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk

        Cls = SynthesizerTrnMs768NSFsid if if_f0 == 1 else SynthesizerTrnMs768NSFsid_nono
        net_g = Cls(*cpt["config"], is_half=False)
        # The posterior encoder isn't used at inference time and weighs a few
        # hundred MB of state we don't need to keep around.
        try: del net_g.enc_q
        except Exception: pass
        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval()
        # Bake weight_norm parametrizations into plain weights BEFORE moving
        # to the inference device. weight_norm keeps the original `weight_g`
        # / `weight_v` params plus a forward hook that recomputes the actual
        # weight on every call. On DirectML those intermediate tensors can
        # straddle CPU and the DML device, manifesting as a hard segfault
        # inside the C kernel (no Python traceback). Removing weight_norm
        # collapses them into a single tensor that .to(device) moves cleanly.
        try:
            # The all-in-one method on SynthesizerTrn* also touches enc_q,
            # which we just deleted — walk submodules ourselves instead.
            net_g.dec.remove_weight_norm()
        except Exception: pass
        try:
            net_g.flow.remove_weight_norm()
        except Exception: pass
        # Move the synthesizer to whichever device the user picked.
        try:
            net_g = net_g.to(self._device)
        except Exception as e:
            # DirectML can refuse certain ops; fall back to CPU rather than
            # blowing up the whole load. Drop any HuBERT we'd loaded on the
            # now-abandoned device, and sync device_pref to "cpu" so the
            # set_device_pref early-return doesn't trap the user — they
            # need to be able to pick another DML adapter to retry.
            self._init_error = (
                f"Move synthesizer to {self._device_label} failed "
                f"({e}) — using CPU."
            )
            self._device, self._device_label = "cpu", "CPU"
            self._device_pref = "cpu"
            self._hubert = None
            net_g = net_g.to("cpu")

        self._net_g = net_g
        self._tgt_sr = tgt_sr
        self._if_f0 = if_f0
        self._version = version
        self._cpt = cpt
        self._current = voice_name
        self._build_pipeline()

    def _build_pipeline(self):
        from rvc_lib.pipeline import Pipeline
        cfg = _Config(device=self._device, is_half=False)
        # Pipeline.__init__ takes lib_dir to locate rmvpe.pt — we use the
        # base dir's parent so `<base_dir>/base_model/rmvpe.pt` resolves.
        # Simplest: stage the file under the layout Pipeline expects.
        bm = self.base_dir / "base_model"
        bm.mkdir(parents=True, exist_ok=True)
        rmvpe = self._ensure_rmvpe()
        rmvpe_link = bm / "rmvpe.pt"
        if not rmvpe_link.exists():
            try: rmvpe_link.symlink_to(rmvpe)
            except Exception:
                shutil.copy2(rmvpe, rmvpe_link)
        self._pipeline = Pipeline(self._tgt_sr, cfg, lib_dir=str(self.base_dir))

    def _ensure_hubert(self):
        if self._hubert is not None:
            return
        from rvc_lib.hubert_adapter import HubertContentExtractor
        try:
            self._hubert = HubertContentExtractor(
                device=self._device, half=False, cache_dir=str(self.hf_cache_dir),
            )
        except Exception:
            # HuBERT choked on DirectML (some ops aren't supported on
            # privateuseone). Pull EVERYTHING back to CPU rather than leaving
            # one half of the graph on DML — otherwise the synth/HuBERT
            # device split crashes in extract_features on the first chunk.
            self._device, self._device_label = "cpu", "CPU"
            self._init_error = (
                "HuBERT n'a pas pu démarrer sur DirectML — bascule de tout "
                "le pipeline sur CPU (ton GPU AMD a probablement un op non "
                "supporté par DirectML pour ce modèle)."
            )
            # Synth and Pipeline were built on the old device — rebuild now.
            if self._net_g is not None:
                try: self._net_g = self._net_g.to("cpu")
                except Exception: pass
            # Force the Pipeline to be re-created with the new device on the
            # next streaming call.
            self._pipeline = None
            self._hubert = HubertContentExtractor(
                device="cpu", half=False, cache_dir=str(self.hf_cache_dir),
            )

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
        """Run RVC inference on `input_path` → `output_path` (.wav). Blocks
        the caller for the duration — meant to be invoked from a worker."""
        with self._lock:
            try:
                self._load_voice(voice_name)
                self._ensure_hubert()
            except Exception as e:
                self._init_error = str(e)
                return False, str(e)

            try:
                import numpy as np
                import librosa
                import soundfile as sf
            except Exception as e:
                return False, (
                    f"Dépendance manquante ({e}). Installe : "
                    "pip install -r requirements-rvc.txt"
                )

            # 1. Load + downsample to 16 kHz mono for HuBERT.
            try:
                audio_16k, _ = librosa.load(str(input_path), sr=16000, mono=True)
            except Exception as e:
                return False, f"Lecture audio échouée: {e}"
            peak = float(np.abs(audio_16k).max() or 1.0)
            if peak > 1.0 / 0.95:
                audio_16k = audio_16k / (peak * 0.95)

            # 2. Find the .index file if one came with this voice.
            idx_dir = self.voices_dir / voice_name
            idx_files = list(idx_dir.glob("*.index"))
            file_index = str(idx_files[0]) if idx_files else ""

            # 3. Run the vendored pipeline.
            try:
                audio_opt = self._pipeline.pipeline(
                    self._hubert,            # mimics fairseq HubertModel
                    self._net_g,             # synthesizer
                    0,                       # speaker id (single-speaker)
                    audio_16k,
                    str(input_path),
                    [0, 0, 0],               # times accumulator (unused)
                    int(f0_up_key),
                    str(f0_method),
                    file_index,
                    float(index_rate),
                    self._if_f0,
                    int(filter_radius),
                    self._tgt_sr,
                    0,                       # resample_sr — keep target
                    float(rms_mix_rate),
                    self._version,
                    float(protect),
                    None,                    # f0_file — none
                )
            except Exception as e:
                import traceback
                return False, f"Pipeline RVC: {e}\n{traceback.format_exc()[-600:]}"

            try:
                out_arr = np.asarray(audio_opt, dtype=np.int16) if audio_opt.dtype != np.int16 else audio_opt
                sf.write(str(output_path), out_arr, self._tgt_sr, subtype="PCM_16")
            except Exception as e:
                return False, f"Sauvegarde WAV échouée: {e}"
        return True, None

    # ─── Streaming inference (for live mic conversion) ─────────────────────
    def prepare_for_streaming(self, voice_name: str) -> tuple[bool, Optional[str]]:
        """Load the voice + hubert once so process_chunk() can be called in a
        tight loop without paying the warm-up cost on every block."""
        with self._lock:
            try:
                self._load_voice(voice_name)
                self._ensure_hubert()
                return True, None
            except Exception as e:
                self._init_error = str(e)
                return False, str(e)

    def streaming_target_sr(self) -> int | None:
        return self._tgt_sr

    last_chunk_error: Optional[str] = None
    last_chunk_peak: float = 0.0

    def process_chunk(
        self,
        audio_16k,  # 1D float32 numpy array, mono at 16 kHz
        *,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.5,
        protect: float = 0.33,
        rms_mix_rate: float = 0.25,
        filter_radius: int = 3,
    ):
        """Run a single Pipeline pass on `audio_16k`. Returns the output at
        the voice's target SR as a float32 numpy array (normalised to [-1,1])
        — or None if the model isn't loaded yet. Records the last error /
        output peak on the instance so the live engine can surface them to
        the UI.
        """
        if self._pipeline is None or self._net_g is None or self._hubert is None:
            return None
        import numpy as np
        idx_dir = self.voices_dir / (self._current or "")
        idx_files = list(idx_dir.glob("*.index")) if idx_dir.exists() else []
        file_index = str(idx_files[0]) if idx_files else ""
        try:
            audio_opt = self._pipeline.pipeline(
                self._hubert,
                self._net_g,
                0,
                audio_16k,
                "live_chunk",
                [0, 0, 0],
                int(f0_up_key),
                str(f0_method),
                file_index,
                float(index_rate),
                self._if_f0,
                int(filter_radius),
                self._tgt_sr,
                0,
                float(rms_mix_rate),
                self._version,
                float(protect),
                None,
            )
        except Exception as e:
            import traceback
            self.last_chunk_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
            return None

        out = audio_opt.astype(np.float32) / 32768.0
        # NaN/Inf can leak from DirectML on unsupported ops — clamp + record
        # the peak so the UI can show "silent / NaN" instead of just nothing.
        if not np.all(np.isfinite(out)):
            self.last_chunk_error = (
                f"sortie non finie ({int((~np.isfinite(out)).sum())} NaN/Inf samples) "
                f"sur {self._device_label} — l'op a probablement échoué silencieusement"
            )
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out
        self.last_chunk_peak = float(np.abs(out).max()) if out.size else 0.0
        # Distinguish "user wasn't speaking, so output is naturally silent"
        # from "model produced zeros despite real input". The first case is
        # the normal idle state — flagging it as an error filled the UI
        # with false positives whenever the mic captured a quiet moment.
        in_peak = float(np.abs(audio_16k).max()) if audio_16k.size else 0.0
        if self.last_chunk_peak < 1e-5 and in_peak > 5e-3:
            self.last_chunk_error = (
                f"sortie ≈ silence (peak={self.last_chunk_peak:.6f}) malgré "
                f"un signal d'entrée (peak={in_peak:.3f}) sur {self._device_label}. "
                "Essaie une autre méthode F0 (CREPE / HARVEST)."
            )
        else:
            # Either we produced real audio, or the user wasn't talking.
            # Either way, clear any previous error so the UI doesn't keep
            # showing a stale message.
            self.last_chunk_error = None
        return out
