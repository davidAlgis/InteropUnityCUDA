<#
.SYNOPSIS
Clean all the temporary files/folders

.DESCRIPTION
Will clean the solution and a set of project directory from temporary files and folder:
In project directory :
- temp folder
- x64 folder
- bin folder
- *.vcxproj file
- *.filters file
- *.user file

In solution directory :
- log.txt file
- compile_commands.json file
- .cache folder
- .vs folder
- .sln file

.PARAMETER projectDirs
An array of directory path that is contains in a string. Each path is separated by a comma.

.PARAMETER solutionDir
The solution directory path to clean.

.EXAMPLE
This commands will clean tessendorf plugin and sample tessendorf
.\buildtools\clean.ps1 -projectDirs ".\SampleBasic, .\PluginInteropUnityCUDA, .\Utilities" -solutionDir ".\"

.NOTES
File Name      : clean.ps1
Author         : David Algis
Prerequisite   : PowerShell v3
Copyright 2023 - Studio Nyx
#>
param (
    [Parameter(Mandatory = $true)]
    [string]$projectDirs,
    [string]$solutionDir
)

if (-not $projectDirs -or -not $solutionDir) {
    Write-Host "Error: One or more arguments are empty."
    exit 1
}

$projectDirArray = $projectDirs -split ","

function CleanProject([string]$projectDir) {
    Write-Host "Clean temporary directory $($projectDir)temp/ and $($projectDir)x64/"
    # $absoluteProjectDir = [System.IO.Path]::GetFullPath($projectDir)
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$($projectDir)x64/"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$($projectDir)temp/"
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$($projectDir)bin/"
    Get-ChildItem -Path $($projectDir) -Recurse -Filter *.vcxproj | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $($projectDir) -Recurse -Filter *.filters | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path $($projectDir) -Recurse -Filter *.user | Remove-Item -Force -ErrorAction SilentlyContinue
}

# Clean projects
foreach ($dir in $projectDirArray) {
    Write-Host "Clean project $dir"
    CleanProject -projectDir $dir
}

# Clean solution
Write-Host "Clean Solution located at $solutionDir"
Remove-Item -ErrorAction SilentlyContinue "$solutionDir\log.txt"
Get-ChildItem -Path $solutionDir -Filter *.sln | Remove-Item -ErrorAction SilentlyContinue
Remove-Item -ErrorAction SilentlyContinue "$solutionDir\compile_commands.json"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$solutionDir\.cache"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$solutionDir\.vs"
