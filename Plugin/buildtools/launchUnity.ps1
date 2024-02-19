. $PSScriptRoot\dependencies.ps1
. $PSScriptRoot\copyBuildToUnity.ps1

$projectPath = "$PSScriptRoot\..\..\InteropUnityCUDA"
Write-Host "Launch Unity interop project at $projectPath..."


& $Env:UNITY_BIN -projectPath "$projectPath"

