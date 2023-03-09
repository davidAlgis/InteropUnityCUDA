using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Utilities;

namespace ActionUnity
{

    public class InteropHandlerSample : InteropHandler
    {
        // the id which will be use in registration of action texture
        private const string _ActionTextureName = "sampleTexture";
        // the id which will be use in registration of action texture array
        private const string _ActionTextureArrayName = "sampleTextureArray";
        // the id which will be use in registration of action vertex buffer
        private const string _ActionVertexBufferName = "sampleVertexBuffer";

        // raw image for texture 
        [SerializeField] private RawImage _rawImageOneTexture;
        [SerializeField] private RawImage _rawImageTextureArray0;
        [SerializeField] private RawImage _rawImageTextureArray1;
        [SerializeField] private ParticlesDrawer _particlesDrawer;
        
        [SerializeField] private int _sizeTexture = 256;
        [SerializeField] private int _sizeBuffer = 256;

        private Texture2D _texture;
        private Texture2DArray _textureArray;
        private Texture2D _textureForDisplay0;
        private Texture2D _textureForDisplay1;
        private ComputeBuffer _computeBuffer;

        /// <summary>
        /// Create a render texture _sizeTexture x _sizeTexture with 4 channel and and set _renderTexture with it
        /// </summary>
        private void CreateTexture()
        {
            _texture = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            _texture.Apply();
            _rawImageOneTexture.texture = _texture;
        }
        
        private void CreateTextureArray()
        {
            _textureArray = new Texture2DArray(_sizeTexture, _sizeTexture, 2, TextureFormat.RGBAFloat, false, true);
            _textureArray.Apply();

            _textureForDisplay0 = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            _textureForDisplay0.Apply();

            _textureForDisplay1 = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            _textureForDisplay1.Apply();
            
            _rawImageTextureArray0.texture = _textureForDisplay0;
            _rawImageTextureArray1.texture = _textureForDisplay1;
        }

        /// <summary>
        /// Create a compute buffer of float4 of size _sizeBuffer
        /// </summary>
        private void CreateBuffer()
        {
            int stride = Marshal.SizeOf(typeof(float4));
            //allocate memory for compute buffer
            _computeBuffer = new ComputeBuffer(_sizeBuffer, stride);
            _particlesDrawer.InitParticlesBuffer(_computeBuffer, _sizeBuffer, 0.1f);
        }

        private void Start()
        {
            InitializeInteropHandler();
        }

        /// <summary>
        /// Create the texture and the buffer. Construct action from them. Register these action in InteropUnityCUDA and
        /// call start function on it
        /// </summary>
        protected override void InitializeActions()
        {
            base.InitializeActions();
            
            if (_particlesDrawer == null)
            {
                Debug.LogError("Set particles drawer in inspector !");
                return;
            }
            
            InitSampleTexture();
            InitSampleTextureArray();
            // InitSampleVertexBuffer();
        }

        private void InitSampleTexture()
        {
            CreateTexture();
            ActionUnitySampleTexture actionUnitySampleTexture = new ActionUnitySampleTexture(_texture);
            RegisterActionUnity(actionUnitySampleTexture, _ActionTextureName);
            CallFunctionStartInAction(_ActionTextureName);
        }

        private void InitSampleTextureArray()
        {
            CreateTextureArray();
            ActionUnitySampleTextureArray actionUnitySampleTextureArray = new ActionUnitySampleTextureArray(_textureArray);
            RegisterActionUnity(actionUnitySampleTextureArray, _ActionTextureArrayName);
            CallFunctionStartInAction(_ActionTextureArrayName);
        }
         
        private void InitSampleVertexBuffer()
        {
            CreateBuffer();
            ActionUnitySampleVertexBuffer actionUnitySampleVertexBuffer = new ActionUnitySampleVertexBuffer(_computeBuffer, _sizeBuffer);
            RegisterActionUnity(actionUnitySampleVertexBuffer, _ActionVertexBufferName);
            CallFunctionStartInAction(_ActionVertexBufferName);
            
        }

        public void Update()
        {
            UpdateInteropHandler();
        }

        /// <summary>
        /// call update function of the two registered actions
        /// </summary>
        protected override void UpdateActions()
        {
            base.UpdateActions();
            CallFunctionUpdateInAction(_ActionTextureName);
            CallFunctionUpdateInAction(_ActionTextureArrayName);
            // CallFunctionUpdateInAction(_ActionVertexBufferName);

            Graphics.CopyTexture(_textureArray, 0, _textureForDisplay0, 0);
            Graphics.CopyTexture(_textureArray, 1, _textureForDisplay1, 0);
        }

        public void OnDestroy()
        {
            OnDestroyInteropHandler();
        }

        /// <summary>
        /// call onDestroy function of the two registered actions
        /// </summary>
        protected override void OnDestroyActions()
        {
            base.OnDestroyActions();
            CallFunctionOnDestroyInAction(_ActionTextureName);
            CallFunctionOnDestroyInAction(_ActionTextureArrayName);
            // CallFunctionOnDestroyInAction(_ActionVertexBufferName);
        }
    }
}
