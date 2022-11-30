using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;
using Utilities;

namespace ActionUnity
{

    public class InteropHandlerSample : InteropHandler
    {
	    private const string _ActionTextureName = "sampleTexture";
	    private const string _ActionVertexBufferName = "sampleVertexBuffer";

	    [SerializeField] private RawImage _rawImage;
	    [SerializeField] private ParticlesDrawer _particlesDrawer;
        
        [SerializeField] private int _sizeTexture = 256;
	    [SerializeField] private int _sizeBuffer = 256;
		

	    
        private RenderTexture _rt;
	    private ComputeBuffer _computeBuffer;
	    private float4[] _cpuArray;

	    private void CreateTexture()
        {
	        _rt = new RenderTexture(_sizeTexture, _sizeTexture, 0
		        , RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
	        {
		        useMipMap = false,
		        autoGenerateMips = false,
		        anisoLevel = 6,
		        filterMode = FilterMode.Trilinear,
		        wrapMode = TextureWrapMode.Clamp,
		        enableRandomWrite = true
	        };

	        print(_rt.Create());
	        _rawImage.texture = _rt;
        }

        private void CreateBuffer()
        {
	        int stride = Marshal.SizeOf(typeof(float4));
		    //allocate memory for compute buffer
		    _computeBuffer = new ComputeBuffer(_sizeBuffer, stride);
		    _particlesDrawer.InitParticlesBuffer(_computeBuffer, _sizeBuffer, 0.1f);
		    _cpuArray = new float4[_sizeBuffer];
        }

        protected override void InitializeActions()
        {
	        base.InitializeActions();
	        
		    if (_particlesDrawer == null)
		    {
			    Debug.LogError("Set particles drawer in inspector !");
			    return;
		    }

		    CreateBuffer();
            CreateTexture();
            ActionUnitySampleTexture actionUnitySampleTexture = new ActionUnitySampleTexture(_rt);
            ActionUnitySampleVertexBuffer actionUnitySampleVertexBuffer = new ActionUnitySampleVertexBuffer(_computeBuffer, _sizeBuffer);
            RegisterActionUnity(actionUnitySampleTexture, _ActionTextureName);
            RegisterActionUnity(actionUnitySampleVertexBuffer, _ActionVertexBufferName);
            CallActionStart(_ActionTextureName);
            CallActionStart(_ActionVertexBufferName);
        }

        protected override void CallUpdateActions()
        {
	        base.CallUpdateActions();
            CallActionUpdate(_ActionTextureName);
            CallActionUpdate(_ActionVertexBufferName);
        }
    }

}