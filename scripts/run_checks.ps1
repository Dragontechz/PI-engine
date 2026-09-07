$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_materials.ps1')
& (Join-Path $PSScriptRoot 'build_input_audio.ps1')
& (Join-Path $PSScriptRoot 'build_physics_tests.ps1')
& (Join-Path $root 'build/material_test.exe')
& (Join-Path $root 'build/input_test.exe')
& (Join-Path $root 'build/audio_test.exe')
& (Join-Path $root 'build/test_sim.exe')
& (Join-Path $root 'build/test_fluid.exe')
& (Join-Path $root 'build/verify.exe')
& (Join-Path $root 'build/verify4.exe')
& (Join-Path $root 'build/verify5.exe')
