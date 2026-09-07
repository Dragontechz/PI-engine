$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'
$raylib = 'C:/Users/kono/Desktop/DEV/C/raylib-5.5/raylib-5.5_win64_mingw-w64'
if (-not (Test-Path -LiteralPath $build)) { New-Item -ItemType Directory -Path $build | Out-Null }
$common = @('-std=c11','-O2','-Wall','-Wextra','-Werror','-I', (Join-Path $raylib 'include'))
$libs = @('-L',(Join-Path $raylib 'lib'),'-lraylib','-lopengl32','-lgdi32','-lwinmm','-lm')
gcc @common -I (Join-Path $root 'src/input') -o (Join-Path $build 'input_test.exe') (Join-Path $root 'tests/input/input_test.c') (Join-Path $root 'src/input/input.c') @libs
if (-not $?) { throw 'Failed to build input_test' }
gcc @common -I (Join-Path $root 'src/audio') -o (Join-Path $build 'audio_test.exe') (Join-Path $root 'tests/audio/audio_test.c') (Join-Path $root 'src/audio/audio.c') @libs
if (-not $?) { throw 'Failed to build audio_test' }
