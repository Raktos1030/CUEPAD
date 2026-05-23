# Q-Pad

Local soundboard with built-in YouTube ripper, for Windows.

Trigger sounds from a grid, route them to a virtual microphone (Discord, OBS, etc. hear them as if you were talking), and rip new sounds straight from YouTube.

## Download

[**Latest release →**](https://github.com/Raktos1030/Q-Pad/releases/latest) — grab `Q-Pad-Setup.exe`, double-click, install. No prerequisites: Python, ffmpeg, portaudio are all bundled.

## Features

### Soundboard
- Auto-generated grid from a sounds folder (`Downloads\Q-Pad Soundboard\` by default)
- Per-sound volume + global volume, real-time
- Dual audio output: a main one (e.g. VB-Cable for Discord/OBS) and an independent local monitor
- Separate mute and volume for the monitor
- Global keyboard shortcuts that work even when Q-Pad isn't focused
- Windows tray: closing the window minimizes to the tray, hotkeys stay live
- Instant **Stop** button for all sounds

### YouTube ripper
- **Original** mode: grabs native audio with no re-encoding (max quality, fast) — supports start/end trimming via stream-copy too
- Re-encoded modes: MP3, WAV, M4A, OPUS — quality 128 / 192 / 320 kbps or Max
- Automatic ID3 metadata and cover art
- Playlist support
- Persistent history

## Using Q-Pad as a virtual microphone

1. Install [VB-Cable](https://vb-audio.com/Cable/) (free)
2. In Q-Pad, pick **CABLE Input** as the main output
3. In Discord / OBS / etc., pick **CABLE Output** as your microphone

The monitor keeps playing on your speakers so you can still hear what you're sending.

## Hotkeys

Configurable per-sound via the ⋮ button on each tile (combos like `Ctrl+1`, `Shift+F2`, etc.). They run in the background — Q-Pad doesn't need to be focused.

## Paths

- Sounds: `Downloads\Q-Pad Soundboard\` (changeable in Settings)
- Settings: `%APPDATA%\Q-Pad\settings.json`

## Local build

```
pip install -r requirements.txt
python main.py
```

The Windows installer is built by `.github/workflows/build.yml`, triggered on `v*` tags (or manually via "Run workflow").
