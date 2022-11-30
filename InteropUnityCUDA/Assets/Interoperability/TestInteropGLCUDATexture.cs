using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class TestInteropGLCUDATexture : MonoBehaviour
{
	[SerializeField] private RawImage _textureDisplay;


#if UNITY_EDITOR || UNITY_STANDALONE
    const string _dllName = "PluginInteropUnityCUDA";
#elif UNITY_IOS
    const string _dllName = "__Internal";
#endif
	
    [DllImport(_dllName)]
    private static extern void SetTextureFromUnity(IntPtr texture, int w, int h);
    
    [DllImport(_dllName)]
    private static extern IntPtr GetRenderEventFunc();
    
    [DllImport(_dllName)]
    private static extern void SetTime(float time);
    
    [DllImport(_dllName)]
    private static extern void CustomUnityPluginUnload();
    
    private RenderTexture _rt;
    
    private int _res = 256;
    
    IEnumerator Start ()
    {
	    CreateTextureAndPassToPlugin();
	    yield break;
	    //Has to be called before eventID 1, because it registered texture in CUDA
	    GL.IssuePluginEvent(GetRenderEventFunc(), 0);
		yield return StartCoroutine("CallPluginAtEndOfFrames");
    }


    private void Update()
    {
	    //SetTime(Time.realtimeSinceStartup);
    }

    private void CreateTextureAndPassToPlugin()
    {
	    if (_textureDisplay == null)
	    {
		    Debug.LogError("Unable to find the canvas to display the function, please complete the serializefield");
	    } 
	    
	    _rt = new RenderTexture(_res, _res, 0
                    , RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                useMipMap = false,
                autoGenerateMips = false,
                anisoLevel = 6,
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Clamp,
                enableRandomWrite = true
            };

		_rt.Create();
	    
		_textureDisplay.texture = _rt;
		
		//Texture2D _rt = new Texture2D(256,256,TextureFormat.ARGB32,false);
		// Set point filtering just so we can see the pixels clearly
		// _rt.filterMode = FilterMode.Point;
		// // Call Apply() so it's actually uploaded to the GPU
		// _rt.Apply();
		// Pass texture pointer to the plugin
		SetTextureFromUnity (_rt.GetNativeTexturePtr(), _rt.width, _rt.height);
		
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

}