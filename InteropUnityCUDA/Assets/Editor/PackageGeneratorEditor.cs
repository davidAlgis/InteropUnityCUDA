using System.Diagnostics;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;
using Debug = UnityEngine.Debug;

namespace Interop.Packager
{
    public class PackageGeneratorEditor : EditorWindow
    {
        private const string _packageScriptLocation = "../Plugin/buildtools/packageUnity.ps1";
        private static PackRequest _currentRequest;

        [MenuItem("Interop/Packager", false, 160)]
        public static void Package()
        {
            var retCode = GeneratePackage();
            if (retCode != 0)
            {
                Debug.LogError("There has been an error while packaging with " + _packageScriptLocation);
                return;
            }

            _currentRequest = Client.Pack("../com.studio-nyx.interop-unity-cuda.1.0.1", "../Artifacts");
            EditorApplication.update += HandlePackOperation;
        }

        private static int GeneratePackage()
        {
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
            process.StandardInput.WriteLine($"& '{_packageScriptLocation}'");

            // Close the input stream to signal that no more input will be provided
            process.StandardInput.Close();

            // Read the output (if needed)
            var output = process.StandardOutput.ReadToEnd();
            var errors = process.StandardError.ReadToEnd();

            // Wait for the process to exit
            process.WaitForExit();
            return process.ExitCode;
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