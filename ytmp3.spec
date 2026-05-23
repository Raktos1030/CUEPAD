# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

wv_datas,    wv_bins,    wv_hidden    = collect_all("webview")
ytdlp_datas, ytdlp_bins, ytdlp_hidden = collect_all("yt_dlp")
mut_datas,   mut_bins,   mut_hidden   = collect_all("mutagen")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[
        (os.path.join("ffmpeg", "ffmpeg.exe"), "."),
        *wv_bins,
        *ytdlp_bins,
        *mut_bins,
    ],
    datas=[
        ("templates", "templates"),
        *wv_datas,
        *ytdlp_datas,
        *mut_datas,
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
        *wv_hidden,
        *ytdlp_hidden,
        *mut_hidden,
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
    name="YT-MP3",
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
    name="YT-MP3",
)
