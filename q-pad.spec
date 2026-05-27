# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

wv_datas,    wv_bins,    wv_hidden    = collect_all("webview")
ytdlp_datas, ytdlp_bins, ytdlp_hidden = collect_all("yt_dlp")
mut_datas,   mut_bins,   mut_hidden   = collect_all("mutagen")
sd_datas,    sd_bins,    sd_hidden    = collect_all("sounddevice")
sci_datas,   sci_bins,   sci_hidden   = collect_all("scipy")
ats_datas,   ats_bins,   ats_hidden   = collect_all("audiotsm")
pn_datas,    pn_bins,    pn_hidden    = collect_all("pynput")
pst_datas,   pst_bins,   pst_hidden   = collect_all("pystray")
pil_datas,   pil_bins,   pil_hidden   = collect_all("PIL")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[
        (os.path.join("ffmpeg", "ffmpeg.exe"), "."),
        *([(os.path.join("ffmpeg", "ffprobe.exe"), ".")]
          if os.path.exists(os.path.join("ffmpeg", "ffprobe.exe")) else []),
        *wv_bins,
        *ytdlp_bins,
        *mut_bins,
        *sd_bins,
        *sci_bins,
        *ats_bins,
        *pn_bins,
        *pst_bins,
        *pil_bins,
    ],
    datas=[
        ("templates", "templates"),
        *([("icon.ico", ".")] if os.path.exists("icon.ico") else []),
        *wv_datas,
        *ytdlp_datas,
        *mut_datas,
        *sd_datas,
        *sci_datas,
        *ats_datas,
        *pn_datas,
        *pst_datas,
        *pil_datas,
    ],
    hiddenimports=[
        "webview",
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "webview.platforms.mshtml",
        "clr",
        "flask",
        "mutagen",
        "mutagen.id3",
        "mutagen.mp4",
        "sounddevice",
        "soundfile",
        "numpy",
        "scipy",
        "scipy.signal",
        "audiotsm",
        "audiotsm.wsola",
        "audiotsm.io.array",
        "effects",
        "live_engine",
        "voice_changer",
        "live_rvc",
        "rvc_lib",
        "rvc_lib.models",
        "rvc_lib.modules",
        "rvc_lib.attentions",
        "rvc_lib.commons",
        "rvc_lib.transforms",
        "rvc_lib.rmvpe",
        "rvc_lib.pipeline",
        "rvc_lib.hubert_adapter",
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._win32",
        "pystray",
        "pystray._win32",
        "PIL.Image",
        "PIL.ImageDraw",
        *wv_hidden,
        *ytdlp_hidden,
        *mut_hidden,
        *sd_hidden,
        *sci_hidden,
        *ats_hidden,
        *pn_hidden,
        *pst_hidden,
        *pil_hidden,
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Q-Pad",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon="icon.ico" if os.path.exists("icon.ico") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Q-Pad",
)
