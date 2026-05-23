# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

# Collect all pywebview files
wv_datas, wv_binaries, wv_hiddenimports = collect_all("webview")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[
        # ffmpeg.exe must be in ./ffmpeg/ffmpeg.exe at build time
        (os.path.join("ffmpeg", "ffmpeg.exe"), "."),
        *wv_binaries,
    ],
    datas=[
        ("templates", "templates"),
        *wv_datas,
    ],
    hiddenimports=[
        "webview",
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "webview.platforms.mshtml",
        "clr",
        "flask",
        "yt_dlp",
        *wv_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
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
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # no terminal window
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
