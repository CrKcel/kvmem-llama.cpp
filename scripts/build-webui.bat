@echo off
setlocal
rem ============================================================
rem  Build the kvmem Web UI (upstream full-ui by default).
rem  中文：构建 kvmem Web UI（默认使用上游完整 full-ui 版本）
rem
rem  Usage: scripts\build-webui.bat [full|light]
rem    full  -> complete upstream llama.cpp UI incl. PWA/manifest
rem            (default; same as v016 kvmem-test --full-ui mode)
rem          中文：完整上游 UI（含 PWA/manifest，默认，同 v016 --full-ui）
rem    light -> lightweight entry points (manifest/PWA excluded)
rem          中文：轻量级入口页面（不含 manifest/PWA）
rem
rem  Output: build-cuda13.4\share\kvmem\ui  (--webui mounts this)
rem          中文：产物输出目录（服务器 --webui 直接挂载此目录）
rem  Requires: git, Python 3, Node.js (npm/npx on PATH).
rem  中文：依赖 git、Python 3、Node.js（npm/npx 需在 PATH）。UI 构建无需 CUDA/MSVC。
rem ============================================================

set "MODE=--full-ui"
if /I "%~1"=="light" set "MODE="
if /I "%~1"=="lightweight" set "MODE="

cd /d "%~dp0\.."
set "PROJECT_DIR=%CD%"
set "OUTPUT_DIR=%PROJECT_DIR%\build-cuda13.4\share\kvmem\ui"
set "SEARCH=build-cuda*"

rem --- tool checks -------------------------------------------------
where git >nul 2>&1 || (echo [ERROR] git not on PATH & pause & exit /b 1)

set "PY=python"
where python >nul 2>&1 || (echo [ERROR] python not on PATH & pause & exit /b 1)

where npm.cmd >nul 2>&1 || (echo [ERROR] npm.cmd not on PATH & pause & exit /b 1)
where npx.cmd >nul 2>&1 || (echo [ERROR] npx.cmd not on PATH & pause & exit /b 1)

echo.
echo Project : %PROJECT_DIR%
echo Mode    : %MODE%  (full = upstream UI, light = lightweight)
echo Output  : %OUTPUT_DIR%
echo.

if not exist "%PROJECT_DIR%\llama.cpp\tools\ui\package.json" (
    echo [ERROR] llama.cpp submodule missing tools\ui - check git submodule.
    pause
    exit /b 1
)

rem --- run the builder ---------------------------------------------
if "%MODE%"=="" (
    python "%PROJECT_DIR%\scripts\build-webui.py" --output "%OUTPUT_DIR%"
) else (
    python "%PROJECT_DIR%\scripts\build-webui.py" --full-ui --output "%OUTPUT_DIR%"
)
if %errorlevel% neq 0 (
    echo === UI BUILD FAILED ===
    pause
    exit /b 1
)

echo.
echo === UI BUILD OK ===
echo   index.html          : %OUTPUT_DIR%\index.html
if exist "%OUTPUT_DIR%\manifest.webmanifest" echo   manifest.webmanifest : %OUTPUT_DIR%\manifest.webmanifest
if exist "%OUTPUT_DIR%\sw.js" echo   sw.js               : %OUTPUT_DIR%\sw.js
echo   ui-build.json mode  :
findstr /c:"mode" "%OUTPUT_DIR%\ui-build.json"

endlocal & exit /b 0