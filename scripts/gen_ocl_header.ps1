$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cl = Join-Path $root 'src/raytracer/gpu_opencl_kernel.cl'
$gen = Join-Path $root 'src/raytracer/gpu_opencl_kernel.h'

$lines = (Get-Content -LiteralPath $cl -Raw) -split "`r?`n"
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('#ifndef GPU_OPENCL_KERNEL_H')
[void]$sb.AppendLine('#define GPU_OPENCL_KERNEL_H')
[void]$sb.AppendLine('/* AUTO-GENERATED from gpu_opencl_kernel.cl by scripts/gen_ocl_header.ps1.')
[void]$sb.AppendLine(' * Edit the .cl file instead and rebuild. */')
[void]$sb.AppendLine('static const char *OCL_SRC =')
foreach ($line in $lines) {
    $e = $line -replace '\\', '\\\\' -replace '"', '\"'
    [void]$sb.AppendLine('    "' + $e + '\n"')
}
[void]$sb.AppendLine('    ;')
[void]$sb.AppendLine('#endif')
Set-Content -LiteralPath $gen -Value $sb.ToString() -Encoding ASCII
Write-Host "Generated $gen"
