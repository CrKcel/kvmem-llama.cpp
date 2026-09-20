#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$Model = (Join-Path $env:LOCALAPPDATA 'KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
    [string]$BuildDir,
    [string]$Gpu = '0',
    [ValidateRange(1, 65535)][int]$Port = 18202,
    [ValidateRange(128, 262144)][int]$Context = 32768,
    [ValidateRange(128, 262144)][int]$Budget = 2048,
    [ValidateRange(128, 262144)][int]$Reserve = 1024,
    [ValidateRange(1, 4096)][int]$Batch = 128,
    [ValidateSet('q8_0', 'q5_0', 'q4_0')][string]$KvType = 'q8_0',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native-process.ps1')
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$BuildDir) { $BuildDir = Join-Path $root 'build-win-bonsai' }
$binary = Join-Path $BuildDir 'bin/llama-kvmem-server.exe'
foreach ($path in @($binary, $Model)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
if ($Budget % 128 -ne 0 -or $Reserve % 128 -ne 0) { throw 'Budget and Reserve must be multiples of 128' }
if ($Budget + $Reserve -gt $Context) { throw 'Budget + Reserve must not exceed Context' }
$binary = (Resolve-Path -LiteralPath $binary).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$serverArgs = @('-m', $Model, '-ngl', '99', '--host', '127.0.0.1', '--port', "$Port",
    '-c', "$Context", '-b', "$Batch", '--ubatch-size', "$Batch", '-n', "$Reserve",
    '--kvmem-budget', "$Budget", '--kvmem-gen-reserve', "$Reserve", '--kvmem-block-tokens', '128',
    '--kv-dtype', $KvType, '--spec-type', 'none', '--kvmem-mtp-state', 'snapshots',
    '--kvmem-query-policy', 'user', '--kvmem-query-replay', 'auto', '--flash-attn', 'on')
if ($DryRun) {
    @{ argv = @($binary) + $serverArgs; environment = @{ CUDA_VISIBLE_DEVICES = $Gpu; CUDA_DEVICE_ORDER = 'PCI_BUS_ID' } } |
        ConvertTo-Json -Depth 5
    return
}
$probe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
$probe.Server.ExclusiveAddressUse = $true
try { $probe.Start() } finally { $probe.Stop() }
$info = New-KVMemProcessInfo $binary $serverArgs
$info.CreateNoWindow = $true
$info.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = $Gpu
$info.EnvironmentVariables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
$child = New-Object System.Diagnostics.Process
$child.StartInfo = $info
$started = $false
try {
    Write-Host "Starting Bonsai without MTP at http://127.0.0.1:$Port/"
    $started = $child.Start()
    while (!$child.WaitForExit(250)) {}
    $code = $child.ExitCode
} finally {
    if ($started -and !$child.HasExited) { $child.Kill(); $child.WaitForExit() }
    $child.Dispose()
}
exit $code
