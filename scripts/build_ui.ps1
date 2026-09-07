$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
$raylib = 'C:/Users/kono/Desktop/DEV/C/raylib-5.5/raylib-5.5_win64_mingw-w64'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
gcc -std=c11 -O2 -Wall -Wextra `
    -o (Join-Path $build 'ui.exe') `
    (Join-Path $root 'src/ui/ui.c') `
    -I (Join-Path $raylib 'include') -L (Join-Path $raylib 'lib') `
    -I 'C:/Users/kono/Desktop/DEV/clay' `
    -lraylib -lopengl32 -lgdi32 -lwinmm -lm
if (-not $?) { throw 'Failed to build ui' }
