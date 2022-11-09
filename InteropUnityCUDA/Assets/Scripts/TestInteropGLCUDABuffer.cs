using System;
using System.Collections;
using System.Runtime.InteropServices;
using TMPro;
using Unity.Mathematics;
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
    private static extern void SetBufferFromUnity(IntPtr buffer, int size);
    
    
    [DllImport(_dllName)]
    private static extern void SetTime(float time);
    
    [DllImport(_dllName)]
    private static extern void UnityShutdown();
    
    [DllImport(_dllName)]
    private static extern IntPtr GetRenderEventFunc();
    
    [SerializeField] private TMP_Text _tmpText;
    [SerializeField] private int _sizeBuffer = 256;
    private ComputeBuffer _computeBuffer;
    private float4[] _cpuArray;
    
    IEnumerator Start ()
    {
    
	    CreateComputeBufferAndPassToPlugin();
	    //Has to be called before eventID 1, because it registered texture in CUDA
		GL.IssuePluginEvent(GetRenderEventFunc(), 2);
		yield return StartCoroutine("CallPluginAtEndOfFrames");
    }
    
    
    private void CreateComputeBufferAndPassToPlugin()
    {
	    int stride = Marshal.SizeOf(typeof(float4));
	    _computeBuffer = new ComputeBuffer(_sizeBuffer, stride);
	    
	    _cpuArray = new float4[_sizeBuffer];
	    for(int i=0; i<_sizeBuffer;i++)
	    {
		    _cpuArray[i] = new float4(i, i*2, i+1, 1);
	    }

	    
	    _computeBuffer.SetData(_cpuArray);
	    
	    for(int i=0; i<_sizeBuffer;i++)
	    {
		    _cpuArray[i] = new float4(0, 1*i, 2*i, 3*i);
	    }
	    
	    SetBufferFromUnity(_computeBuffer.GetNativeBufferPtr(), _sizeBuffer);
		
	}

    private void Update()
    {
	    SetTime(Time.realtimeSinceStartup);
	    _computeBuffer.GetData(_cpuArray);
		_tmpText.text = _cpuArray == null ? "NULL" : _cpuArray[2].ToString();
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
			GL.IssuePluginEvent(GetRenderEventFunc(), 3);
		}
	}

    private void OnDestroy()
    {
	    // UnityShutdown();
	    _computeBuffer.Dispose();
	    
    }
}