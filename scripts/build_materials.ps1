$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
$sources = Get-ChildItem -LiteralPath (Join-Path $root 'src/materials') -Filter '*.c' | Where-Object { $_.Name -ne 'material_test.c' } | ForEach-Object { $_.FullName }
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic `
    -o (Join-Path $build 'material_test.exe') `
    (Join-Path $root 'tests/materials/material_test.c') $sources `
    -I (Join-Path $root 'src/materials') -lm
if (-not $?) { throw 'Failed to build material_test' }
