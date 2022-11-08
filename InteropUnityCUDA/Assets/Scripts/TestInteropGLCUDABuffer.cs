using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class TestInteropGLCUDABuffer : MonoBehaviour
{
#if UNITY_EDITOR || UNITY_STANDALONE
    const string _dllName = "PluginInteropUnityCUDA";
#elif UNITY_IOS
    const string _dllName = "__Internal";
#endif
	
    [DllImport(_dllName)]
    private static extern void SetBufferFromUnity(IntPtr buffer, int size, int stride);
    
    [DllImport(_dllName)]
    private static extern IntPtr GetRenderEventFunc();
    
    [DllImport(_dllName)]
    private static extern void StartLog();
    
    
    [SerializeField] private int _sizeBuffer = 256;
    private ComputeBuffer _computeBuffer;
    
    IEnumerator Start ()
    {
    
	    StartLog();
	    CreateComputeBufferAndPassToPlugin();
	    //Has to be called before eventID 1, because it registered texture in CUDA
		GL.IssuePluginEvent(GetRenderEventFunc(), 2);
		// yield return StartCoroutine("CallPluginAtEndOfFrames");
		yield break;
    }
    
    
    private void CreateComputeBufferAndPassToPlugin()
    {
	    int stride = sizeof(int);

	    _computeBuffer = new ComputeBuffer(_sizeBuffer, stride);
	    print(_computeBuffer.IsValid());
	    
	    SetBufferFromUnity(_computeBuffer.GetNativeBufferPtr(), _sizeBuffer, stride);
		
	}
    
    private IEnumerator CallPluginAtEndOfFrames()
	{
		while (true) {
			// Wait until all frame rendering is done
			yield return new WaitForEndOfFrame();
			
			// Issue a plugin event with arbitrary integer identifier.
			// The plugin can distinguish between different
			// things it needs to do based on this ID.
			// For our simple plugin, it does not matter which ID we pass here.
			GL.IssuePluginEvent(GetRenderEventFunc(), 1);
			
		}
	}

    private void OnDestroy()
    {
	    _computeBuffer.Dispose();
	    
    }
}