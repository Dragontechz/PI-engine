$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
$raylib = 'C:/Users/kono/Desktop/DEV/C/raylib-5.5/raylib-5.5_win64_mingw-w64'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
gcc -std=c11 -O2 -Wall -Wextra -D_DEFAULT_SOURCE `
    -o (Join-Path $build 'verlet3d.exe') `
    (Join-Path $root 'src/physics3d/main3d.c') `
    -I (Join-Path $root 'src/physics3d') `
    -I (Join-Path $raylib 'include') -L (Join-Path $raylib 'lib') `
    -lraylib -lopengl32 -lgdi32 -lwinmm -lm
if (-not $?) { throw 'Failed to build verlet3d' }
