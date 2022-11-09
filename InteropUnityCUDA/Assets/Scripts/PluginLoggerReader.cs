using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// This class will use Utilities.dll to read and print the logs that has been
/// written by the other plugin. 
/// </summary>
class PluginLoggerReader : MonoBehaviour
{
    
    #if UNITY_EDITOR || UNITY_STANDALONE
        const string _dllName = "Utilities";
    #elif UNITY_IOS
        const string _dllName = "__Internal";
    #endif
	
   
    [DllImport(_dllName)]
    private static extern void GetLastLog(out IntPtr data, logType type);

    /// <summary>
    /// Will add this prefix before all logs
    /// </summary>
    [SerializeField]private string _prefixLog = "Plugin - ";
    
    private enum logType
    {
        Info,
        Warning,
        Error
    };

    private void LateUpdate()
    {
        //will print all the plugin logs in unity
        PrintLogInfo();
        PrintLogWarning();
        PrintLogError();
    }

    /// <summary>
    /// Will get from the native plugin a string that sum all log that has been used in the other native plugin
    /// By sum I mean that it'll split all log by '\n'.
    /// It will use a pointer that will get the address of the string.
    /// </summary>
    /// <param name="type">What log we need : info, warning, error.</param>
    /// <returns>a list</returns>
    private List<string> GetLastLog(logType type)
    {
        GetLastLog(out IntPtr logPtr, type);
        List<string> logs = Marshal.PtrToStringAnsi(logPtr)?.Split('\n').ToList();
        //we remove the last one that will be always empty
        return logs?.GetRange(0, logs.Count-1);
    }

    /// <summary>
    /// Will get all the string registered in plugin logger as info
    /// and display them in Unity
    /// </summary>
    private void PrintLogInfo()
    {
        
        var logs = GetLastLog(logType.Info);
        if (logs != null)
        {
            foreach (var log in logs)
            {
                Debug.Log(_prefixLog + log);
            }
        }
    }
    
    /// <summary>
    /// Will get all the string registered in plugin logger as warning
    /// and display them in Unity
    /// </summary>
    private void PrintLogWarning()
    {
        
        var logs = GetLastLog(logType.Warning);
        if (logs != null)
        {
            foreach (var log in logs)
            {
                Debug.LogWarning(_prefixLog + log);
            }
        }
    }
    
    /// <summary>
    /// Will get all the string registered in plugin logger as error
    /// and display them in Unity
    /// </summary>
    private void PrintLogError()
    {
        var logs = GetLastLog(logType.Error);
        if (logs != null)
        {
            foreach (var log in logs)
            {
                Debug.LogError(_prefixLog + log);
            }
        }
    }
}
