"""ONNX export + DirectML inference path for RVC.

PyTorch+DirectML on the RVC synthesizer segfaults reproducibly (see git
log). ONNX Runtime + DirectML — the path w-okada uses on AMD/Intel
hardware — is markedly more stable and faster: the synth runs as a
fully-traced ONNX graph against the DmlExecutionProvider, with no Python
fallback and no torch C-API surface exposed to DirectML's quirks.

This module owns:
 - export_synth_to_onnx(): one-shot conversion of a user-supplied .pth
   into <voice_dir>/<name>.onnx via torch.onnx.export.
 - export_contentvec_to_onnx(): bakes a layer-12 ContentVec ONNX from
   the transformers HubertModel weights the torch path already uses,
   so HuBERT runs on the same backend without a second download path.
 - OnnxRvcSession: cached InferenceSession trio (synth + ContentVec)
   exposing the same shape as the PyTorch path so the live worker
   doesn't care which backend it's driving.

Pitch detection stays on CPU regardless of backend — RMVPE/CREPE on CPU
are fast enough that they don't need GPU, and DirectML support for the
specific pitch ops is patchy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# No reliable public URL hosts the ContentVec ONNX (lj1995 only ships the
# .pt). We export it ourselves from the transformers HubertModel weights
# the rest of the pipeline already uses — one-shot, cached on disk.
CONTENTVEC_HF_MODEL = "lengyue233/content-vec-best"


def export_synth_to_onnx(pth_path: str | Path, onnx_path: str | Path) -> None:
    """Convert an RVC .pth into an ONNX graph that ContentVec features +
    pitch + speaker id can be piped through. One-shot, cached on disk.
    Post-processes the graph to FP16 internally when onnxconverter_common
    is available — IO stays FP32 so callers see no change, but RDNA3 /
    Ampere+ run the inner conv/matmul stack ~1.5-2× faster."""
    import os, shutil, tempfile
    import torch
    from rvc_lib.models_onnx import SynthesizerTrnMsNSFsidM

    cpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    # n_spk slot in the config tuple comes from the embedding table.
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    version = cpt.get("version", "v1")
    if version != "v2":
        raise RuntimeError(
            f"Modèle v{version[1:]} non supporté pour ONNX/DirectML. "
            "Réentraîne ou convertis en v2."
        )

    # Dummy inputs with dynamic time axes so the exported graph accepts
    # any-length chunks at runtime.
    test_phone   = torch.rand(1, 200, 768)
    test_phlen   = torch.tensor([200]).long()
    test_pitch   = torch.randint(size=(1, 200), low=5, high=255)
    test_pitchf  = torch.rand(1, 200)
    test_ds      = torch.LongTensor([0])
    test_rnd     = torch.rand(1, 192, 200)

    net = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=version)
    net.load_state_dict(cpt["weight"], strict=False)
    net.eval()

    # Stage to a temp path so a failed FP16 conversion still leaves the
    # FP32 graph available to fall back to.
    fp32_fd, fp32_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fp32_fd)
    try:
        torch.onnx.export(
            net,
            (test_phone, test_phlen, test_pitch, test_pitchf, test_ds, test_rnd),
            fp32_path,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone":  [1],
                "pitch":  [1],
                "pitchf": [1],
                "rnd":    [2],
            },
            do_constant_folding=True,
            opset_version=17,
            verbose=False,
        )

        _simplify_onnx_in_place(fp32_path, label="synth")
        _try_fp16_then_fp32(fp32_path, str(onnx_path))
    finally:
        try: os.unlink(fp32_path)
        except OSError: pass


def _try_fp16_then_fp32(fp32_path: str, final_path: str) -> None:
    """Synth-specific wrapper around the generic mixed-precision pass."""
    import numpy as _np
    T = 64
    feed = {
        'phone':         _np.random.randn(1, T, 768).astype(_np.float32),
        'phone_lengths': _np.array([T], dtype=_np.int64),
        'pitch':         _np.random.randint(5, 255, size=(1, T)).astype(_np.int64),
        'pitchf':        _np.random.uniform(50, 500, size=(1, T)).astype(_np.float32),
        'ds':            _np.array([0], dtype=_np.int64),
        'rnd':           _np.random.randn(1, 192, T).astype(_np.float32),
    }
    _try_fp16_then_fp32_with_feed(
        src_fp32=fp32_path, dst=final_path, feed=feed, label="synth",
    )


def _try_fp16_then_fp32_with_feed(src_fp32: str, dst: str,
                                  feed: dict, label: str,
                                  op_block_list: list | None = None) -> None:
    """Best-effort mixed-precision conversion with FP32 fallback.

    Two paths:
     - If `op_block_list` is given, delegate to the static converter
       (`convert_float_to_float16`) which accepts op-type blocklists.
       Used for graphs where we know specific op types break the
       auto path (e.g. ['Cast'] for HuBERT's attention which has
       explicit casts for softmax scaling).
     - Otherwise use `auto_convert_mixed_precision`: runs the model in
       FP32 first to capture a reference output, then iteratively tries
       to convert subsets of nodes to FP16, accepting only subsets
       whose output stays within rtol/atol of reference.

    Both paths share the FP32 fallback: any failure → simplified FP32
    graph is saved unchanged so live inference still works.
    """
    if op_block_list:
        _try_static_fp16_then_fp32(src_fp32, dst, list(op_block_list), label)
        return

    import shutil
    try:
        import onnx as _onnx
        from onnxconverter_common.auto_mixed_precision import (
            auto_convert_mixed_precision,
        )
    except ImportError:
        print(f"[VC] onnxconverter-common absent — {label} en FP32 "
              f"(plus lent). Installe : pip install onnxconverter-common",
              flush=True)
        shutil.copy2(src_fp32, dst)
        return

    try:
        model_proto = _onnx.load(src_fp32)
        print(f"[VC] {label} auto-mixed-precision — slow (1-5 min) but "
              f"cached…", flush=True)
        converted = auto_convert_mixed_precision(
            model_proto,
            feed,
            rtol=1e-2,
            atol=1e-2,
            keep_io_types=True,
        )
        _onnx.save(converted, dst)
        size_mb = Path(dst).stat().st_size // (1024 * 1024)
        fp16_count = sum(
            1 for n in converted.graph.node
            for a in n.attribute
            if a.name == 'to' and a.i == _onnx.TensorProto.FLOAT16
        )
        print(f"[VC] {label} ONNX mixed-precision saved — {size_mb} MB, "
              f"~{fp16_count} FP16 ops", flush=True)
    except Exception as e:
        print(f"[VC] {label} mixed-precision failed "
              f"({type(e).__name__}: {e}) — fallback FP32 (simplified)",
              flush=True)
        shutil.copy2(src_fp32, dst)


def _try_static_fp16_then_fp32(src_fp32: str, dst: str,
                               op_block_list: list, label: str) -> None:
    """Static FP16 conversion using `convert_float_to_float16` with
    op-type blocklist + smoke-test gate. Used when the auto path's API
    can't express what we need (no op_block_list parameter) or when a
    known-bad op family needs to be excluded from the start (HuBERT's
    Cast nodes). On any failure → simplified FP32 fallback."""
    import os, shutil
    try:
        import onnx as _onnx
        from onnxconverter_common import float16 as ocnn_fp16
        import onnxruntime as _ort
    except ImportError:
        print(f"[VC] onnxconverter-common absent — {label} en FP32",
              flush=True)
        shutil.copy2(src_fp32, dst)
        return

    try:
        model_proto = _onnx.load(src_fp32)
        print(f"[VC] {label} static-FP16 (op_block={op_block_list})…",
              flush=True)
        converted = ocnn_fp16.convert_float_to_float16(
            model_proto,
            keep_io_types=True,
            op_block_list=op_block_list,
            max_finite_val=65504.0,
            disable_shape_infer=False,
        )
        tmp_path = dst + ".tmp.fp16.onnx"
        _onnx.save(converted, tmp_path)
        try:
            sess = _ort.InferenceSession(
                tmp_path, providers=["CPUExecutionProvider"],
            )
            del sess
        except Exception as e_load:
            try: os.unlink(tmp_path)
            except OSError: pass
            raise RuntimeError(f"ORT rejected FP16 model: {e_load}") from e_load
        shutil.move(tmp_path, dst)
        size_mb = Path(dst).stat().st_size // (1024 * 1024)
        fp16_count = sum(
            1 for n in converted.graph.node
            for a in n.attribute
            if a.name == 'to' and a.i == _onnx.TensorProto.FLOAT16
        )
        print(f"[VC] {label} ONNX static-FP16 saved — {size_mb} MB, "
              f"~{fp16_count} FP16 ops", flush=True)
    except Exception as e:
        print(f"[VC] {label} static-FP16 failed "
              f"({type(e).__name__}: {e}) — fallback FP32 (simplified)",
              flush=True)
        shutil.copy2(src_fp32, dst)


def export_contentvec_to_onnx(dest_path: str | Path,
                              progress_cb=None,
                              cache_dir: str | None = None) -> None:
    """Bake a layer-12 ContentVec ONNX from the HuggingFace HubertModel
    weights the torch path already uses. One-shot, cached on disk —
    ~360 MB after export. We do it locally because no public URL hosts
    the prebuilt ONNX (lj1995 only ships the .pt)."""
    import torch
    try:
        from transformers import HubertModel
    except ImportError as e:
        raise RuntimeError(
            "transformers introuvable — installe avec : "
            "pip install -r requirements-rvc.txt"
        ) from e

    # Bridges HubertModel(input_values=(B, samples)) to the
    # (B, 1, samples) → (B, 768, T) convention OnnxRvcSession already
    # produces/consumes, so the ONNX we bake is shape-compatible with
    # the lj1995 ContentVec ONNX that callers were originally written
    # for. Inline because torch must stay a lazy import at module level.
    class _Wrapper(torch.nn.Module):
        def __init__(self, m, layer):
            super().__init__()
            self.hubert = m
            self.layer = layer
        def forward(self, source):
            source = source.squeeze(1)
            out = self.hubert(
                input_values=source,
                output_hidden_states=True,
                return_dict=True,
            )
            feats = out.hidden_states[self.layer]
            return feats.transpose(1, 2)

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".onnx.part")

    if progress_cb:
        progress_cb(0, 3)

    model = HubertModel.from_pretrained(CONTENTVEC_HF_MODEL, cache_dir=cache_dir)
    model.eval()
    if progress_cb:
        progress_cb(1, 3)

    wrapped = _Wrapper(model, 12)
    wrapped.eval()

    # 1 s of mono audio @ 16 kHz matching the (1, 1, T) shape the call
    # site in OnnxRvcSession.extract_features already produces.
    dummy = torch.zeros(1, 1, 16000, dtype=torch.float32)
    if progress_cb:
        progress_cb(2, 3)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(tmp),
            input_names=["source"],
            output_names=["features"],
            dynamic_axes={
                "source":   {0: "batch", 2: "samples"},
                "features": {0: "batch", 2: "frames"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    _simplify_onnx_in_place(str(tmp), label="contentvec")
    # FP16 the HuBERT graph too — cvec is ~109 ms FP32 on DML at T=70,
    # FP16 typically brings that to ~60-80 ms. Same auto-validated
    # conversion as the synth: only ops whose output stays within
    # rtol/atol of FP32 get converted. Voice-like dummy (sigma=0.1)
    # so the validator sees realistic activations, not noise.
    import numpy as _np
    cvec_feed = {
        "source": (_np.random.randn(1, 1, 16000) * 0.1).astype(_np.float32),
    }
    try:
        # HuBERT's self-attention has explicit Cast nodes for softmax
        # scaling — same pattern that broke the synth conv. Pre-block
        # `Cast` so the converter doesn't try to fold them and crash
        # ORT at load. Everything else (Conv, MatMul, Add, ...) still
        # has a shot at FP16.
        _try_fp16_then_fp32_with_feed(
            src_fp32=str(tmp), dst=str(dest),
            feed=cvec_feed, label="contentvec",
            op_block_list=["Cast"],
        )
    finally:
        try: tmp.unlink()
        except OSError: pass

    if progress_cb:
        progress_cb(3, 3)


def _simplify_onnx_in_place(onnx_path: str, label: str = "graph") -> None:
    """Run onnx-simplifier over an exported graph in place. Fuses
    constants, eliminates redundant Cast/Transpose chains, propagates
    static shape inference. On failure, the original file is kept.
    Each ONNX op is a DML kernel launch — fewer ops = lower per-chunk
    overhead, which is the dominant cost on DirectML."""
    try:
        import onnx as _onnx
        import onnxsim
    except ImportError:
        print(f"[VC] onnx-simplifier absent — {label} non simplifié. "
              f"Installe : pip install onnx-simplifier", flush=True)
        return
    try:
        model = _onnx.load(onnx_path)
        before = len(model.graph.node)
        simplified, ok = onnxsim.simplify(model)
        if not ok:
            print(f"[VC] {label}: onnxsim verification failed — "
                  f"keeping original ({before} nodes)", flush=True)
            return
        after = len(simplified.graph.node)
        _onnx.save(simplified, onnx_path)
        print(f"[VC] {label} simplified: {before} → {after} nodes",
              flush=True)
    except Exception as e:
        print(f"[VC] {label} simplify error ({type(e).__name__}: {e}) — "
              f"keeping original", flush=True)


def _providers_for(device_label: str) -> list:
    """Map our internal device tag to onnxruntime providers, fastest first."""
    label = (device_label or "cpu").lower()
    if "directml" in label or "privateuseone" in label or label.startswith("dml"):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if "cuda" in label:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxRvcSession:
    """Loaded synth ONNX + ContentVec ONNX, exposed with the same call
    shape the rest of the pipeline expects."""

    def __init__(self, synth_onnx: str | Path, contentvec_onnx: str | Path,
                 providers: list, dml_device_index: int | None = None):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(
                "onnxruntime introuvable — installe avec : "
                "pip install onnxruntime-directml"
            ) from e

        provider_options = []
        for p in providers:
            if p == "DmlExecutionProvider" and dml_device_index is not None:
                provider_options.append({"device_id": int(dml_device_index)})
            else:
                provider_options.append({})

        # ORT_ENABLE_ALL turns on layer fusion, constant folding and
        # transpose-elimination — ORT_ENABLE_BASIC (default) leaves a lot
        # of speed on the table for transformer-heavy graphs like HuBERT.
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # DML-specific tuning: DirectML manages its own GPU allocations,
        # ORT's mem_pattern + cpu_mem_arena layer over the top causes
        # redundant work / fragmentation; both faster off. Parallel
        # execution lets independent ops dispatch concurrently, which
        # matters on DML where per-op launch overhead dominates.
        if "DmlExecutionProvider" in providers:
            sess_options.enable_mem_pattern = False
            sess_options.enable_cpu_mem_arena = False
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Synthesizer.
        self.synth = ort.InferenceSession(
            str(synth_onnx),
            sess_options=sess_options,
            providers=providers, provider_options=provider_options,
        )
        # ContentVec / HuBERT.
        self.cvec = ort.InferenceSession(
            str(contentvec_onnx),
            sess_options=sess_options,
            providers=providers, provider_options=provider_options,
        )
        self._synth_input_names = [i.name for i in self.synth.get_inputs()]
        self._cvec_input_name = self.cvec.get_inputs()[0].name
        self.active_providers = self.synth.get_providers()

    def warmup_for_chunk_ms(self, chunk_ms: int) -> None:
        """Pre-trigger DML shader compilation at the exact (T_cvec, T_synth)
        that the live worker will hit, so the first realtime chunk doesn't
        cost multi-second JIT inside the audio callback. DML caches kernels
        per shape, so warming up at the wrong T buys us nothing — we have
        to use the real one (= chunk_ms × 16 / 320 × 2)."""
        import time
        samples = max(320, int(chunk_ms * 16))     # 16 kHz mono
        t_synth = max(2, samples // 320 * 2)        # cvec hop=320, 2× upsample
        t0 = time.monotonic()
        self.cvec.run(None, {
            self._cvec_input_name: np.zeros((1, 1, samples), dtype=np.float32),
        })
        dt_cvec = (time.monotonic() - t0) * 1000.0

        t1 = time.monotonic()
        feed = dict(zip(self._synth_input_names, [
            np.zeros((1, t_synth, 768), dtype=np.float32),  # phone
            np.array([t_synth],          dtype=np.int64),    # phone_lengths
            np.zeros((1, t_synth),       dtype=np.int64),    # pitch
            np.zeros((1, t_synth),       dtype=np.float32),  # pitchf
            np.array([0],                dtype=np.int64),    # ds
            np.zeros((1, 192, t_synth),  dtype=np.float32),  # rnd
        ]))
        self.synth.run(None, feed)
        dt_synth = (time.monotonic() - t1) * 1000.0
        print(f"[VC] ONNX warmup @T_synth={t_synth} "
              f"(chunk={chunk_ms}ms): cvec {dt_cvec:.0f}ms · "
              f"synth {dt_synth:.0f}ms", flush=True)

    def extract_features(self, audio_16k: np.ndarray) -> np.ndarray:
        """Run the ContentVec ONNX over mono 16 kHz audio. Returns
        features shaped (1, T, 768) ready to feed the synthesizer."""
        feats = np.asarray(audio_16k, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats.mean(-1)
        feats = feats.reshape(1, 1, -1)
        out = self.cvec.run(None, {self._cvec_input_name: feats})[0]
        # Upstream returns (1, 768, T) — transpose to (1, T, 768) and stretch
        # by 2 along time (the synth expects a 2x-upsampled feature stream).
        out = out.transpose(0, 2, 1).astype(np.float32)
        out = np.repeat(out, 2, axis=1)
        return out

    def infer(self, phone: np.ndarray, phone_lengths: np.ndarray,
              pitch: np.ndarray, pitchf: np.ndarray,
              ds: np.ndarray, rnd: np.ndarray) -> np.ndarray:
        feed = dict(zip(
            self._synth_input_names,
            [phone, phone_lengths, pitch, pitchf, ds, rnd],
        ))
        audio = self.synth.run(None, feed)[0]
        # ONNX synth returns float in [-1,1]; convert to int16-style scale
        # so downstream code that divides by 32768 still produces float.
        return (audio * 32767.0).astype(np.int16)
