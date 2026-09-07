$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
gcc -std=c11 -O2 -Wall -Wextra -Werror `
    -o (Join-Path $build 'render_world.exe') `
    (Join-Path $root 'src/raytracer/render_world.c') `
    (Join-Path $root 'src/raytracer/tracer.c') `
    (Join-Path $root 'src/raytracer/tracer_scene.c') -lm
if (-not $?) { throw 'Failed to build render_world' }
