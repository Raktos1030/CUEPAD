"""ONNX export + DirectML inference path for RVC.

PyTorch+DirectML on the RVC synthesizer segfaults reproducibly (see git
log). ONNX Runtime + DirectML — the path w-okada uses on AMD/Intel
hardware — is markedly more stable and faster: the synth runs as a
fully-traced ONNX graph against the DmlExecutionProvider, with no Python
fallback and no torch C-API surface exposed to DirectML's quirks.

This module owns:
 - export_synth_to_onnx(): one-shot conversion of a user-supplied .pth
   into <voice_dir>/<name>.onnx via torch.onnx.export.
 - download_contentvec_onnx(): grabs the HuggingFace
   `lj1995/VoiceConversionWebUI` 768-layer-12 ContentVec ONNX into the
   base cache so HuBERT can run on the same backend.
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


CONTENTVEC_URL  = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/vec-768-layer-12.onnx"
CONTENTVEC_SIZE = 343 * 1024 * 1024  # ~343 MB


def export_synth_to_onnx(pth_path: str | Path, onnx_path: str | Path) -> None:
    """Convert an RVC .pth into an ONNX graph that ContentVec features +
    pitch + speaker id can be piped through. One-shot, cached on disk."""
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

    torch.onnx.export(
        net,
        (test_phone, test_phlen, test_pitch, test_pitchf, test_ds, test_rnd),
        str(onnx_path),
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
        dynamic_axes={
            "phone":  [1],
            "pitch":  [1],
            "pitchf": [1],
            "rnd":    [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
    )


def download_contentvec_onnx(dest_path: str | Path,
                             progress_cb=None) -> None:
    """Pull the prebuilt vec-768-layer-12 ContentVec ONNX from HuggingFace."""
    import requests
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".onnx.part")
    with requests.get(CONTENTVEC_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                f.write(chunk)
                written += len(chunk)
                if progress_cb:
                    progress_cb(written, CONTENTVEC_SIZE)
    tmp.replace(dest)


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

        # Synthesizer.
        self.synth = ort.InferenceSession(
            str(synth_onnx), providers=providers, provider_options=provider_options,
        )
        # ContentVec / HuBERT.
        self.cvec = ort.InferenceSession(
            str(contentvec_onnx), providers=providers, provider_options=provider_options,
        )
        self._synth_input_names = [i.name for i in self.synth.get_inputs()]
        self._cvec_input_name = self.cvec.get_inputs()[0].name
        self.active_providers = self.synth.get_providers()

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
