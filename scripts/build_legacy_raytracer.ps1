$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
gcc -std=c11 -O2 -Wall -Wextra `
    -o (Join-Path $build 'legacy_raytracer.exe') `
    (Join-Path $root 'src/raytracer/legacy_raytracer.c') -lm
if (-not $?) { throw 'Failed to build legacy_raytracer' }
