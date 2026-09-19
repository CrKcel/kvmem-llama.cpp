@echo off
setlocal
rem ============================================================
rem  Quick rebuild of llama-kvmem-server.exe only (incremental).
rem  中文：仅增量重建 llama-kvmem-server.exe（不编译其他目标）
rem  Requires an existing configured build tree: build-cuda13.4
rem  (run scripts\build-cuda13.4.bat once first if missing).
rem  中文：需要已配置好的构建目录 build-cuda13.4
rem  （缺失时请先运行 scripts\build-cuda13.4.bat）
rem ============================================================

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "VS_PATH=D:\Program Files\Microsoft Visual Studio\2022\Professional"
set "VS_VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_VCVARS%" (
    echo [ERROR] Visual Studio not found at: %VS_VCVARS%
    pause
    exit /b 1
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo [ERROR] CUDA 13.4 not found at: %CUDA_PATH%
    pause
    exit /b 1
)

call "%VS_VCVARS%" amd64 >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] vcvarsall failed
    pause
    exit /b 1
)

set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"

cd /d "%~dp0\.."
set "PROJECT_DIR=%CD%"

if not exist "%PROJECT_DIR%\build-cuda13.4\CMakeCache.txt" (
    echo [ERROR] build tree not configured: build-cuda13.4\CMakeCache.txt missing.
    echo         Run scripts\build-cuda13.4.bat once first.
    pause
    exit /b 1
)

rem Warn if the server is currently running (exe is locked, link would fail).
rem 中文：若服务器正在运行则 exe 被锁定，链接会失败，此处先行告警。
tasklist /fi "IMAGENAME eq llama-kvmem-server.exe" 2>nul | find /i "llama-kvmem-server.exe" >nul
if %errorlevel% equ 0 (
    echo [WARN] llama-kvmem-server.exe is RUNNING - the binary is locked.
    echo        Stop the server first, then re-run this script.
    pause
    exit /b 1
)

echo.
echo CUDA: %CUDA_PATH%
echo Building llama-kvmem-server.exe (incremental)...
echo.

ninja -C "%PROJECT_DIR%\build-cuda13.4" llama-kvmem-server
if %errorlevel% neq 0 (
    echo === BUILD FAILED ===
    pause
    exit /b 1
)

for %%F in ("%PROJECT_DIR%\build-cuda13.4\bin\llama-kvmem-server.exe") do (
    echo === BUILD OK ===
    echo exe: %%~fF
    echo size: %%~zF bytes  time: %%~tF
)

echo.
pause
endlocal & exit /b 0