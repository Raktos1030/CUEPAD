@echo off
setlocal enabledelayedexpansion
title Build Cuepad

echo.
echo  ================================
echo   Build Cuepad Desktop App
echo  ================================
echo.

:: -- 1. Dependances Python --
echo [1/4] Installation des dependances Python...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (echo ERREUR: pip install -r requirements.txt a echoue & pause & exit /b 1)
pip install pyinstaller --quiet
if %errorlevel% neq 0 (echo ERREUR: pip install pyinstaller a echoue & pause & exit /b 1)

:: -- 2. FFmpeg --
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
  "$ffprobe = Get-ChildItem -Path 'ffmpeg\tmp' -Filter 'ffprobe.exe' -Recurse | Select-Object -First 1;" ^
  "if ($ffprobe) { Copy-Item $ffprobe.FullName 'ffmpeg\ffprobe.exe' }" ^
  "Remove-Item 'ffmpeg\tmp' -Recurse -Force;" ^
  "Remove-Item $zip -Force;" ^
  "Write-Host '      FFmpeg OK.'"

if not exist "ffmpeg\ffmpeg.exe" (
    echo ERREUR: ffmpeg.exe introuvable apres telechargement.
    pause & exit /b 1
)

:: -- 3. PyInstaller --
:build
echo [3/4] Build PyInstaller...
pyinstaller cuepad.spec --clean --noconfirm
if %errorlevel% neq 0 (echo ERREUR: PyInstaller a echoue & pause & exit /b 1)

:: -- 4. Inno Setup --
echo [4/4] Recherche de Inno Setup...
set ISCC=""
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe"      set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"

if %ISCC%=="" (
    echo       Inno Setup non installe -- setup.exe skipped.
    echo       L'app est quand meme disponible dans dist\Cuepad\
    goto :done
)

mkdir release 2>nul
%ISCC% installer.iss
if %errorlevel% neq 0 (echo ERREUR: Inno Setup a echoue & pause & exit /b 1)
echo       Installeur cree dans release\Cuepad-Setup.exe

:done
echo.
echo  ================================
echo   Build termine !
echo  ================================
echo.
echo  - App portable  : dist\Cuepad\Cuepad.exe
if exist "release\Cuepad-Setup.exe" echo  - Installeur    : release\Cuepad-Setup.exe
echo.
pause
