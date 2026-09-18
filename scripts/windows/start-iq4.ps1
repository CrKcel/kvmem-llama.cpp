#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$global:LASTEXITCODE = 0
& (Join-Path $PSScriptRoot 'start-server.ps1') -Recipe iq4 @args
exit $LASTEXITCODE
