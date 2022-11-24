using System;
using System.Collections;
using System.Runtime.InteropServices;
using TMPro;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Utilities;

public class ActionTOTO : Action
{
#if UNITY_EDITOR || UNITY_STANDALONE
    const string _dllName = "TotoDLL";
#endif
	
    [DllImport(_dllName)]
    private static extern IntPtr createActionToto(int key);
	
    [DllImport(_dllName)]
    private static extern void setTextureActionToto(IntPtr actionPtr, IntPtr texturePtr);
    
	private ActionToto() {}

    static ActionTOTO create(int key) {
		ActionToto toto;
		toto._actionPtr = createActionToto(key);
	} 
    
    IEnumerator Start ()
    {

	    if (_particlesDrawer == null)
	    {
		    Debug.LogError("Set particles drawer in inspector !");
		    yield break;
	    }
	    
	    CreateComputeBufferAndPassToPlugin();
	    //Has to be called before eventID 1, because it registered texture in CUDA
		GL.IssuePluginEvent(GetRenderEventFunc(), 2);
		yield return StartCoroutine("CallPluginAtEndOfFrames");
    }
    
    
    private void CreateComputeBufferAndPassToPlugin()
    {
	    int stride = Marshal.SizeOf(typeof(float4));
	    //allocate memory for compute buffer
	    _computeBuffer = new ComputeBuffer(_sizeBuffer, stride);
	    _particlesDrawer.InitParticlesBuffer(_computeBuffer, _sizeBuffer, 0.1f);
	    _cpuArray = new float4[_sizeBuffer];
	    
	    //send pointer on native pointer to register it in graphics api and 
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
			GL.IssuePluginEvent(GetRenderEventFunc(), 3);
		}
	}

    private void OnDestroy()
    {
	    // UnityShutdown();
	    _computeBuffer.Dispose();
	    
    }
}