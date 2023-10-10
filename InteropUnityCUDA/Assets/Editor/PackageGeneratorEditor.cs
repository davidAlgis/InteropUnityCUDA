using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Interop.Packager
{
    public class PackageGeneratorEditor : EditorWindow
    {
        private const string _packageScriptLocation = "../Plugin/buildtools/packageUnity.ps1";
        private static string _packageSrc = "../com.studio-nyx.interop-unity-cuda.";
        private static PackRequest _currentRequest;

        [MenuItem("Interop/Packager", false, 160)]
        public static void Package()
        {
            var versionEnvVar = "%INTEROP_UNITY_CUDA_VERSION%";
            var version = Environment.ExpandEnvironmentVariables(versionEnvVar);
            if (version == versionEnvVar)
            {
                var defaultVersion = "1.0.1";
                Debug.LogWarning(
                    "There has been an error while fetching the version of the package. Please make sure you have " +
                    "launch the project with the powershell script launchUnity.ps1. By default, we will set the version to " +
                    defaultVersion);
                version = defaultVersion;
            }


            var retCode = GeneratePackage();
            if (retCode != 0)
            {
                Debug.LogError("There has been an error while packaging with " + _packageScriptLocation);
                return;
            }

            _packageSrc += version;

            if (Directory.Exists(_packageSrc) == false)
            {
                Debug.LogError("Cannot find source folder at location " + _packageSrc + " abort !");
                return;
            }

            // generate the tarball
            _currentRequest = Client.Pack(_packageSrc, "../Artifacts");
            EditorApplication.update += HandlePackOperation;
        }

        /// <summary>
        ///     Call the script <see cref="_packageScriptLocation" /> to generate the package
        /// </summary>
        /// <returns>return code of script.</returns>
        private static int GeneratePackage()
        {
            var pathArg = Path.Combine(Application.dataPath, "../" + _packageSrc);
            Debug.Log("Package at " + pathArg);
            // Create a new process start info
            var psi = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            // Start the process
            var process = new Process
            {
                StartInfo = psi
            };

            process.Start();

            // Pass the script to the PowerShell process
            process.StandardInput.WriteLine($"& '{_packageScriptLocation}' {pathArg}");
            // Close the input stream to signal that no more input will be provided
            process.StandardInput.Close();
            // Read the output (if needed)
            var output = process.StandardOutput.ReadToEnd();
            // var errors = process.StandardError.ReadToEnd();

            // Wait for the process to exit
            process.WaitForExit();

            // Log output and errors
            Debug.Log("Output script : \n" + output);

            var retCode = process.ExitCode;
            // Close the process
            process.Close();
            return retCode;
        }

        private static void HandlePackOperation()
        {
            if (_currentRequest.IsCompleted)
            {
                var operationResult = _currentRequest?.Result;
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


                    // Consume the packing queue
                    EditorApplication.update -= HandlePackOperation;
                }
                else
                {
                    Debug.Log("Packaging operation: Handling of an incorrect request. Aborting.");
                }
            }
            else if (_currentRequest.Status == StatusCode.InProgress)
            {
                TickFakeProgressbar("Packaging as a tarball in progress",
                    "Waiting for Unity to finish...");
            }
            else
            {
                // Avoid calling this for nothing
                EditorApplication.update -= HandlePackOperation;
                EditorUtility.ClearProgressBar();
            }
        }

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