$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
$raylib = 'C:/Users/kono/Desktop/DEV/C/raylib-5.5/raylib-5.5_win64_mingw-w64'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }

# Keep the embedded OpenCL source in sync with the .cl kernel file.
$cl = Join-Path $root 'src/raytracer/gpu_opencl_kernel.cl'
$gen = Join-Path $root 'src/raytracer/gpu_opencl_kernel.h'
$need_gen = -not (Test-Path -LiteralPath $gen)
if (-not $need_gen) {
    $need_gen = (Get-Item -LiteralPath $cl).LastWriteTime -gt (Get-Item -LiteralPath $gen).LastWriteTime
}
if ($need_gen) {
    & (Join-Path $PSScriptRoot 'gen_ocl_header.ps1')
}

# NOTE: gpu_renderer.c / gpu_opencl.c must NOT get -march=native: the AVX2
# codegen crashes inside GpuRenderer_Render on this machine. Plain -O3 is
# stable, and these files are not compute-hot on the CPU side anyway.
$base = @('-std=c11', '-O3', '-ffast-math', '-fno-math-errno', '-Wall', '-Wextra', '-Werror', '-fopenmp')
$simd = @('-march=native', '-mtune=native')
$inc = @('-I', (Join-Path $raylib 'include'), '-I', (Join-Path $root 'src/raytracer'))

$objs = @()
foreach ($f in @('world_viewer', 'tracer', 'tracer_scene', 'pitsr')) {
    $obj = Join-Path $build "$f.o"
    gcc @base @simd -c (Join-Path $root "src/raytracer/$f.c") -o $obj @inc
    if (-not $?) { throw "Failed to compile $f" }
    $objs += $obj
}
foreach ($f in @('gpu_renderer', 'gpu_opencl')) {
    $obj = Join-Path $build "$f.o"
    gcc @base -c (Join-Path $root "src/raytracer/$f.c") -o $obj @inc
    if (-not $?) { throw "Failed to compile $f" }
    $objs += $obj
}

gcc $objs -o (Join-Path $build 'world_viewer.exe') -fopenmp `
    -L (Join-Path $raylib 'lib') `
    -lraylib -lopengl32 -lgdi32 -lwinmm -lm
if (-not $?) { throw 'Failed to link world_viewer' }
