<#
.SYNOPSIS
This script builds specified projects or the whole solution using MsBuild.

.DESCRIPTION
This PowerShell script is used to build specific projects or the entire solution
using the MsBuild tool. It takes in parameters for projects, action, and configuration
and constructs and executes the appropriate MsBuild commands.

.PARAMETER projects
A string containing the project name or solution to build. Must be one of this possibilities (store in $possibleProjectValues) : 
("PluginInteropUnityCUDA", "SampleBasic"
    , "Utilities")
or "solution" or "sln" or "all" if you want to build the whole solution

.PARAMETER action
A string indicating the action to perform (e.g., build, rebuild, clean).

.PARAMETER configuration
A string indicating the build configuration (e.g., release, debug).

.EXAMPLE
.\BuildProjects.ps1 -projects @("PluginInteropUnityCUDA", "SampleBasic") -action "build" -configuration "release"
This example command builds the specified projects in release configuration.

.NOTES
File Name      : BuildProjects.ps1
Author         : David Algis
Prerequisite   : PowerShell v3 and MsBuild
Copyright 2023 - Studio Nyx
#>
param(
    [string]$project,
    [string]$action,
    [string]$configuration,
    [string]$msbuildPath = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin"
)


. $PSScriptRoot\dependencies.ps1

$version = $Env:INTEROP_UNITY_CUDA_VERSION
# For non dll project, write NO in description
[string[][]]$possibleProjectValues = @(
    @("PluginInteropUnityCUDA", "Contains a library with the core of interoperability between Unity and CUDA."),
    @("SampleBasic", "NO"),
    @("Utilities", "Contains the source of a library that contains utilities and common tools for all the other library and executable.")
)


Write-Host "Launch $action of $project..."-ForegroundColor DarkMagenta

$currentDir = Get-Location

$buildSolution = $false

# Check if the project name is in the list of possible project values
$isValidProject = $false



if ($action -eq "build" -or $action -eq "rebuild") {
    if ($configuration -ne "debug" -and $configuration -ne "release" -and $configuration -ne "") {
        Write-Warning "The configuration argument '$configuration' is not a valid configuration for the action : $action. Exiting the program."
        Exit
    }
}
else {
    if ($action -ne "clean") {
        Write-Warning "The action argument '$action' is not a valid action. Exiting the program."
        Exit
    }
}

if ($configuration -ne "") {
    $configuration = "/p:Configuration=$configuration"
}

if ($project -eq "sln" -or $project -eq "solution" -or $project -eq "all") {
    $buildSolution = $true
}
else {
    foreach ($possibleProject in $possibleProjectValues) {
        if ($possibleProject[0].ToLower() -eq $project.ToLower()) {
            $isValidProject = $true
            # to be sure it has the correct cases
            $project = $possibleProject[0]
            $projectDescription = "$($possibleProject[1]) $configuration)"
            break
        }
    }

    if (-not $isValidProject) {
        Write-Warning "The project name '$project' is not a valid project. Exiting the program."
        Exit
    }

}



Set-Location -Path "$PSScriptRoot\..\bin\"




# if the user want to build the whole solutions
if ($buildSolution) {


    $msbuildArguments = @(
        "PluginInteropUnitySolution.sln",
        "/t:$action",
        "/m",
        "$configuration"
    )
    Write-Host "Execute command : MsBuild $msbuildArguments"
    & $msbuildPath\MsBuild.exe $msbuildArguments
        
    Set-Location -Path $currentDir
    exit
}
   


$msbuildArguments = @(
    "-target:$project",
    "/t:$action",
    "/m",
    "$configuration"
)
Write-Host "Execute command : MsBuild $msbuildArguments"
& $msbuildPath\MsBuild.exe $msbuildArguments

Set-Location -Path $currentDir
