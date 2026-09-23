#Requires -Version 5.1
[CmdletBinding()]
param([int]$Jobs = 4, [switch]$Mtp)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$reference = Join-Path $root 'build-win-prism-reference'
if ($Mtp) { $reference = Join-Path $root 'build-win-prism-mtp' }
$source = Join-Path $reference 'source'
$cuda = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9'
function Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed: $LASTEXITCODE" }
}
if ((& git -C "$root/llama.cpp" rev-parse HEAD) -ne '9a9394a895b96003ca842a6041cb28ac49a108f7') {
    throw 'Unexpected Prism pin'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
& (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PATH = "$cuda/bin;$env:PATH"
if (!(Test-Path -LiteralPath $source)) {
    New-Item -ItemType Directory -Path $reference -Force | Out-Null
    Checked git @('-C', "$root/llama.cpp", 'archive', '--format=zip', "--output=$reference/prism.zip", 'HEAD')
    Expand-Archive -LiteralPath "$reference/prism.zip" -DestinationPath $source
}
if ($Mtp) {
    $patch = Join-Path $root 'patches/prism-bonsai-mtp-embedding.patch'
    $relativeSource = 'build-win-prism-mtp/source'
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & git -C $root apply --reverse --check "--directory=$relativeSource" $patch 2>$null
        $applied = $LASTEXITCODE -eq 0
    } finally { $ErrorActionPreference = $savedPreference }
    if (!$applied) {
        Checked git @('-C', $root, 'apply', '--check', "--directory=$relativeSource", $patch)
        Checked git @('-C', $root, 'apply', "--directory=$relativeSource", $patch)
    }
    if (!(Select-String -LiteralPath "$source/src/models/qwen35.cpp" -SimpleMatch 'hadamard_inverses->find(tok_embd_w)' -Quiet)) {
        throw 'MTP embedding patch was not applied'
    }
}
# Build GGML from the pristine archived Prism source. Never reuse optimized KVMem libraries.
$wrapper = @'
cmake_minimum_required(VERSION 3.20)
project(prism_reference LANGUAGES C CXX)
add_subdirectory(source)
'@
[IO.File]::WriteAllText("$reference/CMakeLists.txt", $wrapper)
Checked cmake @('-S', $reference, '-B', "$reference/build", '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
    '-DCMAKE_CUDA_ARCHITECTURES=120a-real',
    '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_USE_SYSTEM_GGML=OFF', '-DGGML_CUDA=ON',
    '-DGGML_NATIVE=OFF', '-DGGML_CUDA_FA_ALL_QUANTS=ON',
    "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc.exe",
    "-DCUDAToolkit_ROOT=$cuda", '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON',
    '-DLLAMA_BUILD_SERVER=ON', '-DLLAMA_BUILD_UI=OFF', '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_OPENSSL=OFF',
    '-DLLAMA_BUILD_COMMIT=9a9394a', '-DLLAMA_BUILD_NUMBER=1')
Checked cmake @('--build', "$reference/build", '--parallel', "$Jobs", '--target', 'llama-server', 'llama-bench')
