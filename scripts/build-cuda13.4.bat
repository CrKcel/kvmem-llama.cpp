@echo off
setlocal

rem =============================================================
rem  kvmem-llama.cpp full build script (CUDA 13.4 / Ninja / VS2022)
rem  中文：kvmem-llama.cpp 完整构建脚本（CUDA 13.4 / Ninja / VS2022）
rem  Usage: scripts\build-cuda13.4.bat   (configure + build all)
rem  用法：运行本脚本即完成 CMake 配置与全量编译
rem =============================================================
echo ============================================
echo   kvmem-llama.cpp Build Script (CUDA 13.4)
echo ============================================

cd /d "%~dp0\.."
set "PROJECT_DIR=%CD%"

rem CUDA toolkit and MSVC toolchain locations - adjust to your machine.
rem 中文：以下为本机 CUDA 工具链与 Visual Studio 路径，请按需修改。
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
set "BUILD_DIR=build-cuda13.4"
set "NINJA_STATUS=[%%f/%%t] "

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo.
echo CUDA: %CUDA_PATH%
echo Building: %BUILD_DIR%
echo.

echo Running CMake configure...
cmake -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120a" ^
    -DGGML_CUDA=ON ^
    -DGGML_CUDA_FA_ALL_QUANTS=ON ^
    -DKVMEM_BUILD_LLAMA=ON ^
    -DLLAMA_KVMEM=ON ^
    -DLLAMA_KVMEM_ROOT="%PROJECT_DIR%"

if %errorlevel% neq 0 (
    echo === CONFIGURE FAILED ===
    pause
    exit /b 1
)

echo.
echo Running build...
cmake --build "%BUILD_DIR%" --config Release -j

if %errorlevel% neq 0 (
    echo === BUILD FAILED ===
    pause
    exit /b 1
) else (
    echo === BUILD OK ===
)

pause
