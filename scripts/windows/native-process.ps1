# Windows PowerShell 5.1 loses embedded quotes with native @args invocation.
# Encode argv using the Windows CRT rules and launch without cmd.exe/a shell.
function ConvertTo-KVMemArgument([AllowEmptyString()][string]$Value) {
    '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
}

function New-KVMemProcessInfo([string]$Binary, [string[]]$Arguments) {
    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $Binary
    $info.UseShellExecute = $false
    $info.Arguments = (($Arguments | ForEach-Object { ConvertTo-KVMemArgument $_ }) -join ' ')
    return $info
}
