<#
.NOTES
File Name      : configureProjects.ps1
Author         : David Algis
Prerequisite   : PowerShell v3 and MsBuild
Copyright 2023 - Studio Nyx
#>
param (
    [string]$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
)

Write-Host "Configure Plugin Interop Unity CUDA..."-ForegroundColor DarkMagenta


$currentDir = Get-Location

# Get the parent directory of the script's location
$parentDir = Split-Path -Path $PSScriptRoot -Parent

# Define the path for the bin folder
$binFolder = Join-Path -Path $parentDir -ChildPath "bin"

# Check if the bin folder exists, if not, create it
if (-not (Test-Path -Path $binFolder -PathType Container)) {
    New-Item -Path $binFolder -ItemType Directory | Out-Null
}

# Navigate to the parent directory
Set-Location -Path $binFolder

# Execute the cmake command
$cmakeCommand = "cmake .. -DCMAKE_GENERATOR_TOOLSET=`"cuda=$cudaPath`" -D CMAKE_EXPORT_COMPILE_COMMANDS=ON"
Invoke-Expression -Command $cmakeCommand

Set-Location -Path $currentDir
