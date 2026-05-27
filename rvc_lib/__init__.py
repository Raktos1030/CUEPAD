"""Vendored RVC inference code.

Sources:
- models.py / modules.py / attentions.py / commons.py / transforms.py:
  RVC-Project/Retrieval-based-Voice-Conversion-WebUI synthesizer architectures
  (VITS-style generator + posterior encoder + flow + residual coupling),
  MIT-licensed, copied from the `rvc-python` PyPI package.
- rmvpe.py: RMVPE pitch detector used by RVC, MIT-licensed, also from
  rvc-python.

What was changed vs. upstream:
- Imports rewritten to flat layout (`from rvc_lib import …`).
- Intel ipex / XPU bootstrap removed from rmvpe.py (it pulled in
  `rvc_python.modules.ipex`, which we don't ship).
- HuBERT loading is NOT vendored — see hubert_adapter.py, which uses
  HuggingFace `transformers.HubertModel` to bypass fairseq entirely.
"""
