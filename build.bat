@echo off
setlocal enabledelayedexpansion
title Build YT-MP3

echo.
echo  ================================
echo   Build YT-MP3 Desktop App
echo  ================================
echo.

:: ── 1. Dependances Python ─────────────────────────────────────────────────────
echo [1/4] Installation des dependances Python...
pip install pywebview pyinstaller yt-dlp flask --quiet
if %errorlevel% neq 0 (echo ERREUR: pip install a echoue & pause & exit /b 1)

:: ── 2. FFmpeg ─────────────────────────────────────────────────────────────────
echo [2/4] Verification de ffmpeg...
if exist "ffmpeg\ffmpeg.exe" (
    echo       ffmpeg deja present, on skip.
    goto :build
)

echo       Telechargement de ffmpeg ^(~80 Mo^)...
mkdir ffmpeg 2>nul

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip';" ^
  "$zip = 'ffmpeg\ffmpeg.zip';" ^
  "Write-Host '      Download en cours...';" ^
  "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;" ^
  "Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing;" ^
  "Write-Host '      Extraction...';" ^
  "Expand-Archive -Path $zip -DestinationPath 'ffmpeg\tmp' -Force;" ^
  "$ffexe = Get-ChildItem -Path 'ffmpeg\tmp' -Filter 'ffmpeg.exe' -Recurse | Select-Object -First 1;" ^
  "Copy-Item $ffexe.FullName 'ffmpeg\ffmpeg.exe';" ^
  "Remove-Item 'ffmpeg\tmp' -Recurse -Force;" ^
  "Remove-Item $zip -Force;" ^
  "Write-Host '      FFmpeg OK.'"

if not exist "ffmpeg\ffmpeg.exe" (
    echo ERREUR: ffmpeg.exe introuvable apres telechargement.
    echo Telechargez manuellement ffmpeg.exe et placez-le dans le dossier ffmpeg\
    pause & exit /b 1
)

:: ── 3. PyInstaller ────────────────────────────────────────────────────────────
:build
echo [3/4] Build PyInstaller...
pyinstaller ytmp3.spec --clean --noconfirm
if %errorlevel% neq 0 (echo ERREUR: PyInstaller a echoue & pause & exit /b 1)

:: ── 4. Inno Setup (optionnel) ─────────────────────────────────────────────────
echo [4/4] Recherche de Inno Setup...
set ISCC=""
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe"      set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"

if %ISCC%=="" (
    echo       Inno Setup non installe — setup.exe skipped.
    echo       L'app est quand meme disponible dans dist\YT-MP3\
    goto :done
)

mkdir release 2>nul
%ISCC% installer.iss
if %errorlevel% neq 0 (echo ERREUR: Inno Setup a echoue & pause & exit /b 1)
echo       Installeur cree dans release\YT-MP3-Setup.exe

:: ── Done ──────────────────────────────────────────────────────────────────────
:done
echo.
echo  ================================
echo   Build termine !
echo  ================================
echo.
echo  - App portable  : dist\YT-MP3\YT-MP3.exe
if exist "release\YT-MP3-Setup.exe" echo  - Installeur    : release\YT-MP3-Setup.exe
echo.
pause
