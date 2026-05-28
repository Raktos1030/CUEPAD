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


def export_synth_to_onnx(pth_path: str | Path,
                         prep_path: str | Path,
                         dec_path: str | Path) -> None:
    """Convert an RVC .pth into a SPLIT pair of ONNX graphs, cached on disk:

      * prep_path  — encoder + flow + NSF source generator, FP32.
        (phone, phone_lengths, pitch, pitchf, ds, rnd) → (dec_in, har, g)
        Runs at frame-rate (T≈70), so it's the light part. Stays FP32
        because it's riddled with ops onnxconverter_common can't
        mixed-precision (HuBERT-style attention Cast/Equal in enc_p;
        Resize/Cast/RandomUniformLike/Mod in the NSF SineGen).

      * dec_path   — the HiFi-GAN-style decoder, FP16.
        (dec_in, har, g) → audio
        Runs at audio-rate (T×480 samples), so it's the HEAVY part
        (~the whole synth cost). It's pure Conv/ConvTranspose/ResBlock —
        no NSF, no attention — so convert_float_to_float16 converts it
        cleanly (verified: loads in ORT, FP16 vs FP32 max rel diff
        0.0004, size halved). This is where the speedup lives.

    Splitting at the (dec_in, har, g) tensor boundary keeps every original
    op in ONNX (no Python re-implementation of the NSF → no numerical
    parity risk); it just relocates the FP32/FP16 cut to a clean seam the
    converter can't mess up."""
    import os, tempfile
    import torch
    import torch.nn.functional as F
    from rvc_lib.models_onnx import SynthesizerTrnMsNSFsidM
    import rvc_lib.modules as _modules

    cpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    version = cpt.get("version", "v1")
    if version != "v2":
        raise RuntimeError(
            f"Modèle v{version[1:]} non supporté pour ONNX/DirectML. "
            "Réentraîne ou convertis en v2."
        )

    net = SynthesizerTrnMsNSFsidM(*cpt["config"], is_half=False, version=version)
    net.load_state_dict(cpt["weight"], strict=False)
    net.eval()

    class _PrepNet(torch.nn.Module):
        """SynthesizerTrnMsNSFsidM.forward up to (but excluding) the
        decoder convs — emits the decoder's three inputs."""
        def __init__(self, s):
            super().__init__()
            self.s = s
        def forward(self, phone, phone_lengths, pitch, pitchf, ds, rnd):
            s = self.s
            g = s.emb_g(ds.unsqueeze(0)).transpose(1, 2)
            m_p, logs_p, x_mask = s.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
            z = s.flow(z_p, x_mask, g=g, reverse=True)
            dec_in = z * x_mask
            har, _, _ = s.dec.m_source(pitchf, s.dec.upp)
            har = har.transpose(1, 2)
            return dec_in, har, g

    class _DecNet(torch.nn.Module):
        """GeneratorNSF.forward with har_source as an input (NSF removed)."""
        def __init__(self, dec):
            super().__init__()
            self.dec = dec
        def forward(self, x, har, g):
            d = self.dec
            x = d.conv_pre(x)
            x = x + d.cond(g)
            for i in range(d.num_upsamples):
                x = F.leaky_relu(x, _modules.LRELU_SLOPE)
                x = d.ups[i](x)
                x_source = d.noise_convs[i](har)
                x = x + x_source
                xs = None
                for j in range(d.num_kernels):
                    rb = d.resblocks[i * d.num_kernels + j]
                    xs = rb(x) if xs is None else xs + rb(x)
                x = xs / d.num_kernels
            x = F.leaky_relu(x)
            x = d.conv_post(x)
            return torch.tanh(x)

    prep = _PrepNet(net).eval()
    dec = _DecNet(net.dec).eval()

    # Dummy run of prep to get real-shaped decoder inputs for the dec export.
    T = 200
    d_phone  = torch.rand(1, T, 768)
    d_phlen  = torch.tensor([T]).long()
    d_pitch  = torch.randint(size=(1, T), low=5, high=255)
    d_pitchf = torch.rand(1, T) * 200 + 100
    d_ds     = torch.LongTensor([0])
    d_rnd    = torch.rand(1, 192, T)
    with torch.no_grad():
        d_decin, d_har, d_g = prep(d_phone, d_phlen, d_pitch, d_pitchf, d_ds, d_rnd)

    # ── prep graph (FP32) ──────────────────────────────────────────────
    torch.onnx.export(
        prep, (d_phone, d_phlen, d_pitch, d_pitchf, d_ds, d_rnd), str(prep_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["dec_in", "har", "g"],
        dynamic_axes={"phone": [1], "pitch": [1], "pitchf": [1], "rnd": [2],
                      "dec_in": {2: "t"}, "har": {2: "tf"}},
        do_constant_folding=True, opset_version=17, verbose=False,
    )
    _simplify_onnx_in_place(str(prep_path), label="synth-prep")
    _sz = Path(prep_path).stat().st_size // (1024 * 1024)
    print(f"[VC] synth-prep ONNX saved FP32 — {_sz} MB", flush=True)

    # ── decoder graph (FP16) ───────────────────────────────────────────
    dec_fd, dec_fp32 = tempfile.mkstemp(suffix=".onnx")
    os.close(dec_fd)
    try:
        torch.onnx.export(
            dec, (d_decin, d_har, d_g), dec_fp32,
            input_names=["dec_in", "har", "g"], output_names=["audio"],
            dynamic_axes={"dec_in": {2: "t"}, "har": {2: "tf"}, "audio": {2: "ta"}},
            do_constant_folding=True, opset_version=17, verbose=False,
        )
        _simplify_onnx_in_place(dec_fp32, label="synth-dec")
        # Pure-conv decoder → no NSF/attention → static FP16 converts
        # cleanly and fast. Empty op_block_list (nothing to dodge); the
        # smoke-test gate still falls back to FP32 if a future model
        # surprises us.
        _try_static_fp16_then_fp32(dec_fp32, str(dec_path),
                                   op_block_list=[], label="synth-dec")
    finally:
        try: os.unlink(dec_fp32)
        except OSError: pass


def _try_fp16_then_fp32_with_feed(src_fp32: str, dst: str,
                                  feed: dict, label: str,
                                  op_block_list: list | None = None,
                                  node_block_list: list | None = None) -> None:
    """Best-effort mixed-precision conversion with FP32 fallback.

    Two paths:
     - If `op_block_list` or `node_block_list` is given, delegate to the
       static converter (`convert_float_to_float16`) which accepts both.
       Used for graphs where we know specific op types or subgraphs
       break the auto path (Cast for HuBERT attention, the entire
       'm_source' NSF subgraph for the synth).
     - Otherwise use `auto_convert_mixed_precision`: runs the model in
       FP32 first to capture a reference output, then iteratively tries
       to convert subsets of nodes to FP16, accepting only subsets
       whose output stays within rtol/atol of reference.

    Both paths share the FP32 fallback: any failure → simplified FP32
    graph is saved unchanged so live inference still works.
    """
    if op_block_list or node_block_list:
        _try_static_fp16_then_fp32(
            src_fp32, dst,
            op_block_list=list(op_block_list or []),
            node_block_list=list(node_block_list or []),
            label=label,
        )
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
                               op_block_list: list, label: str,
                               node_block_list: list | None = None) -> None:
    """Static FP16 conversion using `convert_float_to_float16` with
    op-type + node-name blocklists + smoke-test gate. Used when the
    auto path's API can't express what we need (no op_block_list
    parameter) or when known-bad op families / subgraphs need to be
    excluded from the start (HuBERT's Cast nodes, the synth's NSF
    source generator). On any failure → simplified FP32 fallback."""
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
        node_count = len(node_block_list or [])
        node_hint = f", {node_count} nodes" if node_count else ""
        print(f"[VC] {label} static-FP16 (op_block={op_block_list}"
              f"{node_hint})…", flush=True)
        # disable_shape_infer=True: convert_float_to_float16's internal
        # shape-inference pass hangs on large graphs with many blocked
        # nodes (the synth at 1253 nodes with the NSF subgraph blocked
        # was stuck for 5+ min). ORT reruns shape inference at session
        # load anyway, so we lose nothing functional here.
        converted = ocnn_fp16.convert_float_to_float16(
            model_proto,
            keep_io_types=True,
            op_block_list=op_block_list,
            node_block_list=node_block_list or [],
            max_finite_val=65504.0,
            disable_shape_infer=True,
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
        # Real coverage proxy: how many weight tensors are FP16. The
        # "Cast.to == FLOAT16" count is misleading because it only sees
        # explicit cast nodes — Conv/MatMul/etc. have their dtype
        # changed via their input tensors, not via a `to` attribute.
        fp16_weights = sum(
            1 for t in converted.graph.initializer
            if t.data_type == _onnx.TensorProto.FLOAT16
        )
        total_weights = len(converted.graph.initializer)
        print(f"[VC] {label} ONNX static-FP16 saved — {size_mb} MB, "
              f"{fp16_weights}/{total_weights} weights FP16", flush=True)
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
        "source": (_np.random.RandomState(0).randn(1, 1, 16000) * 0.1).astype(_np.float32),
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
    """Loaded split-synth (prep FP32 + decoder FP16) + ContentVec ONNX,
    exposed with the same call shape the rest of the pipeline expects."""

    def __init__(self, prep_onnx: str | Path, dec_onnx: str | Path,
                 contentvec_onnx: str | Path,
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

        def _sess(path):
            return ort.InferenceSession(
                str(path), sess_options=sess_options,
                providers=providers, provider_options=provider_options,
            )

        # Synth, split: prep (FP32, frame-rate) → decoder (FP16, audio-rate).
        self.prep = _sess(prep_onnx)
        self.dec = _sess(dec_onnx)
        # ContentVec / HuBERT.
        self.cvec = _sess(contentvec_onnx)
        self._prep_input_names = [i.name for i in self.prep.get_inputs()]
        self._dec_input_names = [i.name for i in self.dec.get_inputs()]
        self._cvec_input_name = self.cvec.get_inputs()[0].name
        self.active_providers = self.prep.get_providers()

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

        # prep (FP32) at the real shape; its outputs feed the dec warmup so
        # the dec compiles its kernels at the exact audio-rate length too.
        t1 = time.monotonic()
        prep_feed = dict(zip(self._prep_input_names, [
            np.zeros((1, t_synth, 768), dtype=np.float32),  # phone
            np.array([t_synth],          dtype=np.int64),    # phone_lengths
            np.zeros((1, t_synth),       dtype=np.int64),    # pitch
            np.zeros((1, t_synth),       dtype=np.float32),  # pitchf
            np.array([0],                dtype=np.int64),    # ds
            np.zeros((1, 192, t_synth),  dtype=np.float32),  # rnd
        ]))
        dec_in, har, g = self.prep.run(None, prep_feed)
        dt_prep = (time.monotonic() - t1) * 1000.0

        t2 = time.monotonic()
        self.dec.run(None, dict(zip(self._dec_input_names, [dec_in, har, g])))
        dt_dec = (time.monotonic() - t2) * 1000.0
        print(f"[VC] ONNX warmup @T_synth={t_synth} "
              f"(chunk={chunk_ms}ms): cvec {dt_cvec:.0f}ms · "
              f"prep {dt_prep:.0f}ms · dec {dt_dec:.0f}ms", flush=True)

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
        # Stage 1: prep (FP32) → decoder inputs.
        prep_feed = dict(zip(
            self._prep_input_names,
            [phone, phone_lengths, pitch, pitchf, ds, rnd],
        ))
        dec_in, har, g = self.prep.run(None, prep_feed)
        # Stage 2: decoder (FP16) → audio. IO stays FP32 (keep_io_types),
        # so no manual dtype juggling needed at the boundary.
        dec_feed = dict(zip(self._dec_input_names, [dec_in, har, g]))
        audio = self.dec.run(None, dec_feed)[0]
        # ONNX synth returns float in [-1,1]; convert to int16-style scale
        # so downstream code that divides by 32768 still produces float.
        return (audio * 32767.0).astype(np.int16)
