function AbsolutePath($path) {
    # Strips out any relative path modifiers like '..' and '.'
    $abs_path = [System.IO.Path]::GetFullPath($path)

    return $abs_path
}

$Env:INTEROP_UNITY_CUDA_PLUGIN_ROOT = AbsolutePath ([System.IO.Path]::GetDirectoryName( $MyInvocation.MyCommand.Definition) + "\..")
$Env:INTEROP_UNITY_CUDA_UNITY_PROJECT_ROOT = AbsolutePath ([System.IO.Path]::GetDirectoryName( $MyInvocation.MyCommand.Definition) + "\..\..\InteropUnityCUDA")

# To complete in function of your configuration
$Env:INTEROP_UNITY_CUDA_VERSION = "1.0.1"
$Env:UNITY_BIN = "C:\Program Files\Unity\Hub\Editor\2021.3.23f1\Editor\Unity.exe"
