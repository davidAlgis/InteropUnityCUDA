$projectPath = "$PSScriptRoot\..\..\InteropUnityCUDA"
$pathBin = "$PSScriptRoot\..\bin"
$pathPlugin = "$projectPath\Assets\Runtime\Plugin"
Write-Host "Copy binary into the Unity project..."
robocopy  $pathBin $pathPlugin /s


