function AbsolutePath($path) {
    # Strips out any relative path modifiers like '..' and '.'
    $abs_path = [System.IO.Path]::GetFullPath($path)

    return $abs_path
}

$Env:INTEROP_UNITY_CUDA_VERSION = "1.0.1"
$Env:INTEROP_UNITY_CUDA_PLUGIN_ROOT = AbsolutePath ([System.IO.Path]::GetDirectoryName( $MyInvocation.MyCommand.Definition) + "\..")
$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT = AbsolutePath ([System.IO.Path]::GetDirectoryName( $MyInvocation.MyCommand.Definition) + "\..\..\InteropUnityCUDA")
$Env:MSBUILD = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin"

# Write-Host "$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT"
# $Env:INTEROP_UNITY_CUDA_PLUGIN_TARGET = "\Assets\Runtime\Plugin"
$Env:UNITY_2021_3_17 = "C:\Program Files\Unity\Hub\Editor\2021.3.23f1\Editor"
$Env:UNITY_BIN = "$Env:UNITY_2021_3_17\Unity.exe"
