using System;
using System.Collections.Generic;
using System.IO;
using Toolkit3DE.UnityEditor;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;

namespace Interop.Packager
{
    public class PackageGenerator
    {
        public static readonly string PackageName = "com.studio-nyx.interop-unity-cuda";
        private readonly Queue<PackData> _packRequests;

        private Request _currentRequest;
        private string _outputDirectory;

        // public event Action<PackOperationResult> EventPackagingEnded = delegate { };

        // Toolkit tarball
        private string _packagesContainerPath;
        private string _packageTemplatePath;
        private List<string> _thirdPartiesPackages;

        public PackageGenerator()
        {
            _packRequests = new Queue<PackData>();

            _thirdPartiesPackages = new List<string>
            {
                "com.unity.collections@1.3.1",
                "com.unity.mathematics@1.2.6"
            };

            _packageTemplatePath =
                Path.GetFullPath(SafeCombine(ProjectEnvironment.UnityDataPath, "../../" + PackageName));
            SetWorkingDirectory(ProjectEnvironment.BuildPath);
        }

        public event Action<PackOperationResult> EventPackagingEnded = delegate { };


        /// <summary>
        ///     Setup the package folder container where all generated data will be added.
        /// </summary>
        public void SetWorkingDirectory(string outputDirectory)
        {
            _outputDirectory = outputDirectory;

            _packagesContainerPath = SafeCombine(_outputDirectory, "toolkit_package");
        }


        /// <summary>
        ///     Pack created package inside a tarball to be ready to import into another project.
        /// </summary>
        public void Pack()
        {
            // Package and packager
            var toolkitPackage = SafeCombine(_packagesContainerPath, PackageName);

            _packRequests.Enqueue(new PackData { Source = toolkitPackage, Target = _packagesContainerPath });

            AsyncPack();
        }

        private void AsyncPack()
        {
            if (_packRequests.Count > 0)
            {
                var data = _packRequests.Dequeue();
                if (_currentRequest == null || _currentRequest.IsCompleted)
                {
                    // Handle request tracking manually
                    if (Directory.Exists(data.Source))
                    {
                        _currentRequest = Client.Pack(data.Source, data.Target);
                        EditorApplication.update += HandlePackOperation;
                    }
                    else
                    {
                        Debug.Log("Packing operation: Package has to be created/generated first.");
                    }
                }
                else
                {
                    Debug.Log("Packing operation: Already in process. Wait for it to end.");
                }
            }
        }


        private void HandlePackOperation()
        {
            if (_currentRequest.IsCompleted)
            {
                var operationResult = (_currentRequest as PackRequest)?.Result;
                if (operationResult != null)
                {
                    switch (_currentRequest.Status)
                    {
                        case StatusCode.Failure:
                            Debug.Log(_currentRequest.Error.message);
                            break;
                        case StatusCode.Success:
                            Debug.Log(
                                $"Packaging operation: Tarball created at {operationResult.tarballPath}.");
                            break;
                    }

                    // Avoid calling this for nothing
                    EditorUtility.ClearProgressBar();

                    // Notify subscribers of the end of packing before moving on to the next one
                    OnTarballCreationEnded();

                    // Consume the packing queue
                    EditorApplication.update -= HandlePackOperation;
                    AsyncPack();
                }
                else
                {
                    Debug.Log("Packaging operation: Handling of an incorrect request. Aborting.");
                }
            }
            else if (_currentRequest.Status == StatusCode.InProgress)
            {
                UnityEditorUtilities.TickFakeProgressbar("Packaging as a tarball in progress",
                    "Waiting for Unity to finish...");
            }
            else
            {
                // Avoid calling this for nothing
                EditorApplication.update -= HandleInstallOperation;
                EditorUtility.ClearProgressBar();
            }
        }


        private void OnTarballCreationEnded()
        {
            EventPackagingEnded.Invoke((_currentRequest as PackRequest)?.Result);
        }


        private void HandleInstallOperation()
        {
            if (_currentRequest.IsCompleted)
            {
                var operationResult = (_currentRequest as AddRequest)?.Result;
                if (operationResult != null)
                {
                    switch (_currentRequest.Status)
                    {
                        case StatusCode.Failure:
                            Debug.Log(_currentRequest.Error.message);
                            break;
                        case StatusCode.Success:
                            Debug.Log(
                                $"Package installation operation: Package {operationResult.displayName} (by {operationResult.author} added.");
                            break;
                    }

                    // Avoid calling this for nothing
                    EditorApplication.update -= HandleInstallOperation;
                }

                else
                {
                    Debug.Log(
                        "Package installation operation: Handling of an incorrect request. Aborting.");
                }
            }
            else
            {
                // Avoid calling this for nothing
                EditorApplication.update -= HandleInstallOperation;
            }
        }

        /// <summary>
        ///     Extensions method providing similar behavior as Pathy.Combine but without
        ///     the hard dependency of 3DEClient dll.
        /// </summary>
        public static string SafeCombine(string path1, string path2)
        {
            if (path1 == null) return path2;
            if (path2 == null) return path1;

            return path1.Trim().TrimEnd(Path.DirectorySeparatorChar)
                   + Path.DirectorySeparatorChar
                   + path2.Trim().TrimStart(Path.DirectorySeparatorChar);
        }


        private struct PackData
        {
            public string Source;
            public string Target;
        }
    }


    /// <summary>
    ///     Extend Unity Editor tools.
    /// </summary>
    public static class UnityEditorUtilities
    {
        /// <summary>
        ///     Let user be aware that Unity is doing something and that editor is not freezing by faking a progress bar ...
        ///     This method should be called on each EditorApplication.update (on the method register as a callback).
        ///     When this progress bar is no longer needed EditorUtility.ClearProgressBar must be called to let Unity be aware.
        /// </summary>
        public static void TickFakeProgressbar(string title, string info)
        {
            var fakeProgress = (float)(EditorApplication.timeSinceStartup % 100.0);
            fakeProgress *= 0.01F;
            EditorUtility.DisplayProgressBar(title, info, fakeProgress);
        }
    }
}