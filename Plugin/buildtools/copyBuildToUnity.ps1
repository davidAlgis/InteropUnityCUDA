$projectPath = "$PSScriptRoot\..\..\InteropUnityCUDA"
$pathBuildDebug = "$PSScriptRoot\..\Build\Debug"
$pathBuildRelease = "$PSScriptRoot\..\Build\Release"
$pathPluginDebug = "$projectPath\Assets\Runtime\Plugin\Debug"
$pathPluginRelease = "$projectPath\Assets\Runtime\Plugin\Release"
Write-Host "Copy binary into the Unity project..."
robocopy  $pathBuildDebug $pathPluginDebug /s
robocopy  $pathBuildRelease $pathPluginRelease /s


