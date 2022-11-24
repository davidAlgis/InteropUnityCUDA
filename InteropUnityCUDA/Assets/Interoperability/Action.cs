using System;
using System.Collections;
using System.Runtime.InteropServices;
using TMPro;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Utilities;

public abstract class Action : MonoBehaviour
{
#if UNITY_EDITOR || UNITY_STANDALONE
    const string _dllPluginInteropUC = "PluginInteropUnityCUDA";
#endif
	
    [DllImport(_dllPluginInteropUC)]
    private static extern void SetBufferFromUnity(IntPtr buffer, int size);

    [DllImport(_dllPluginInteropUC)]
    private static extern void SetTextureFromUnity(IntPtr texture, int w, int h);
    
    protected int _key = -1;

    public int Key { get => _key; }

    
    protected IntPtr _actionPtr = null;
    
    public IntPtr ActionPtr { get => _actionPtr; }

    IEnumerator Start ()
    {

    }

    private void OnDestroy()
    {
	    // UnityShutdown();
	    _computeBuffer.Dispose();
	    
    }
}