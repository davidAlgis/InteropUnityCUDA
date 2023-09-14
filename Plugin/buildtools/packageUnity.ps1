<#
.SYNOPSIS
This script generate a folder that can be used as a unity package.

.PARAMETER targetFolder
Location of the folder where the Unity package will be generated, it automatically add the version to it.
eg. com.studio-nyx.interop-unity-cuda. will become ..\\com.studio-nyx.interop-unity-cuda.1.0.1

.EXAMPLE
.\packageUnity.ps1 "..\\..\\com.studio-nyx.interop-unity-cuda."
will result in a folder .com.studio-nyx.interop-unity-cuda.1.0.1

.NOTES
File Name      : packageUnity.ps1
Author         : David Algis
Prerequisite   : PowerShell v3
Copyright 2023 - Studio Nyx
#>
param(
    [string]$targetFolder
)

# Include version from dependencies.ps1
. $PSScriptRoot\dependencies.ps1
$version = $Env:INTEROP_UNITY_CUDA_VERSION
$targetFolder = "$targetFolder$version"

# Check if the target folder exists
if (-not (Test-Path -Path $targetFolder -PathType Container)) {
    Write-Host "Target folder does not exist. Creating folder: $targetFolder"
    New-Item -Path $targetFolder -ItemType Directory -Force
}


# Define the content of package.json
$jsonContent = @{
    "name"        = "com.studio-nyx.interop-unity-cuda"
    "version"     = $version
    "displayName" = "Interop Unity CUDA"
    "unity"       = "2021.1"
    "description" = "Demonstrate interoperability between Unity Engine and CUDA."
    "license"     = "MIT"
    "keywords"    = @("GPU", "CUDA", "OpenGL", "DX11", "Native-Plugin", "interoperability")
    "author"      = @{
        "name"  = "David Algis"
        "email" = "david.algis@tutamail.com"
        "url"   = "https://github.com/davidAlgis"
    }
} | ConvertTo-Json -Depth 4

# Create package.json in the target folder
$jsonContent | Set-Content -Path (Join-Path -Path $targetFolder -ChildPath 'package.json') -Force
Write-Host "package.json created in $targetFolder"

# Copy and paste content from another folder to the target folder
$sourceFolder = "$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT\Assets"  

Write-Host "Copy $sourceFolder to $targetFolder..."
Copy-Item -Path $sourceFolder\* -Destination $targetFolder -Recurse -Force

