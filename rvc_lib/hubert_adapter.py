"""HuBERT / ContentVec content extractor for RVC, fairseq-free.

Upstream RVC loads `hubert_base.pt` via fairseq's `checkpoint_utils`. That
forces a fairseq install, which on Windows + Python 3.11 means dragging in
omegaconf 2.0.6 (broken metadata), hydra-core 1.0.x, antlr4-python3-runtime
4.8, and a C++ build of fairseq's `libbleu` extension that needs MSVC Build
Tools — a nightmare we'd rather not impose.

Same model weights are published in HuggingFace transformers format under
`lengyue233/content-vec-best`. We load them with `transformers.HubertModel`
and expose a tiny adapter that mimics the slice of the fairseq interface
the RVC pipeline actually calls: `extract_features(source, padding_mask,
output_layer)` returning a tuple whose first element is the hidden state.

Notes:
- v2 RVC models (the common case in 2025) use the layer-12 hidden state
  directly and need no projection. We support these.
- v1 RVC models use the layer-9 hidden state followed by a `final_proj`
  Linear layer (768 → 256). `lengyue233/content-vec-best` doesn't ship that
  projection, so v1 conversion will be rejected with a clear error rather
  than silently producing garbage. Users with v1 models should convert
  them to v2 (most modern RVC training pipelines do).
"""
from __future__ import annotations

import torch


HUBERT_MODEL_ID = "lengyue233/content-vec-best"


class HubertContentExtractor:
    """Thin fairseq-compatible wrapper around transformers HubertModel."""

    def __init__(self, device: str = "cpu", half: bool = False,
                 model_id: str = HUBERT_MODEL_ID, cache_dir: str | None = None):
        try:
            from transformers import HubertModel
        except Exception as e:
            raise RuntimeError(
                "transformers indisponible — installe avec : "
                "pip install -r requirements-rvc.txt"
            ) from e
        # First time: downloads ~360 MB from HuggingFace into cache_dir.
        self.model = HubertModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = self.model.to(device)
        if half:
            self.model = self.model.half()
        self.model.eval()
        self.device = device
        self._half = half

    @torch.no_grad()
    def extract_features(self, source: torch.Tensor,
                         padding_mask: torch.Tensor | None = None,
                         output_layer: int = 12):
        """Return RVC-shaped features.

        `source` is (batch, samples) at 16 kHz, mono. We pass it through the
        encoder asking for every hidden state and pick the requested layer.
        fairseq's output_layer is 1-indexed in the same way transformers'
        `hidden_states` ends up indexed (states[N] = output after N encoder
        blocks; states[0] is the conv-feature projection).
        """
        if source.dtype != self.model.dtype:
            source = source.to(self.model.dtype)
        # attention_mask: 1 = keep, 0 = ignore. We get padding_mask in the
        # fairseq convention (True = padded), so flip it.
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long()
        out = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        layer = max(0, min(len(out.hidden_states) - 1, int(output_layer)))
        feats = out.hidden_states[layer]
        # Pipeline expects `(features_tuple, …)` — first element is taken.
        return (feats,)

    # `final_proj` is intentionally missing — see the module docstring.
    final_proj = None
