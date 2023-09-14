using System;

namespace Toolkit3DE.UnityEditor
{
    public struct Dependency
    {
        public string Name;
        public string Path;
    }

    public static class ProjectEnvironment
    {
        // Misc
        public static readonly string Version = Environment.ExpandEnvironmentVariables("%INTEROP_UNITY_CUDA_VERSION%");

        // Paths
        public static readonly string BuildPath = Environment.ExpandEnvironmentVariables("%TOOLKIT_BUILD_PATH%");
        public static readonly string UnityDataPath = UnityEngine.Application.dataPath;
    }
}
