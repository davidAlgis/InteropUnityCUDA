$projectPath = "$PSScriptRoot\..\..\InteropUnityCUDA"
$pathBinDebug = "$PSScriptRoot\..\bin\Debug"
$pathBinRelease = "$PSScriptRoot\..\bin\Release"
$pathPluginDebug = "$projectPath\Assets\Runtime\Plugin\Debug"
$pathPluginRelease = "$projectPath\Assets\Runtime\Plugin\Release"
Write-Host "Copy binary into the Unity project..."
robocopy  $pathBinDebug $pathPluginDebug /s
robocopy  $pathBinRelease $pathPluginRelease /s


