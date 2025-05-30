using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;
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

        // the id which will be use in registration of action struct buffer
        private const string _ActionStructBufferName = "sampleStructBuffer";

        // raw image for texture 
        [SerializeField] private RawImage _rawImageOneTexture;
        [SerializeField] private RawImage _rawImageTextureArray0;
        [SerializeField] private RawImage _rawImageTextureArray1;
        [SerializeField] private ParticlesDrawer _particlesDrawer;

        [SerializeField] private int _sizeTexture = 256;
        [SerializeField] private int _sizeBuffer = 256;

        private Texture2D _textureForDisplay0;
        private Texture2D _textureForDisplay1;

        public Texture2D Texture { get; private set; }

        public Texture2DArray TextureArray { get; private set; }

        public ComputeBuffer ComputeVertexBuffer { get; private set; }
        public ComputeBuffer ComputeStructBuffer { get; private set; }


        public void Start()
        {
            InitializeInteropHandler();
        }

        public void Update()
        {
            UpdateInteropHandler();
        }

        public void OnDestroy()
        {
            OnDestroyInteropHandler();
            ComputeVertexBuffer.Dispose();
            ComputeStructBuffer.Dispose();
        }

        /// <summary>
        ///     Create a render texture _sizeTexture x _sizeTexture with 4 channel and and set _renderTexture with it
        /// </summary>
        private void CreateTexture()
        {
            Texture = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            Texture.Apply();
            if (_rawImageOneTexture != null)
                _rawImageOneTexture.texture = Texture;
            else
                Debug.LogWarning(
                    "If you want to visualize the change please filled the _rawImageOneTexture field in InteropHandlerSample monobehavior.");
        }

        private void CreateTextureArray()
        {
            TextureArray = new Texture2DArray(_sizeTexture, _sizeTexture, 2, TextureFormat.RGBAFloat, false, true);
            TextureArray.Apply();

            _textureForDisplay0 = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            _textureForDisplay0.Apply();

            _textureForDisplay1 = new Texture2D(_sizeTexture, _sizeTexture, TextureFormat.RGBAFloat, false, true);
            _textureForDisplay1.Apply();

            if (_rawImageTextureArray0 != null)
                _rawImageTextureArray0.texture = _textureForDisplay0;
            else
                Debug.LogWarning(
                    "If you want to visualize the change please filled the _rawImageTextureArray0 field in InteropHandlerSample monobehavior.");

            if (_rawImageTextureArray1 != null)
                _rawImageTextureArray1.texture = _textureForDisplay1;
            else
                Debug.LogWarning(
                    "If you want to visualize the change please filled the _rawImageTextureArray1 field in InteropHandlerSample monobehavior.");
        }

        /// <summary>
        ///     Create a compute buffer of float4 of size _sizeBuffer
        /// </summary>
        private void CreateBuffer()
        {
            var stride = Marshal.SizeOf(typeof(float4));
            //allocate memory for compute buffer
            ComputeVertexBuffer = new ComputeBuffer(_sizeBuffer, stride);

            if (_particlesDrawer != null)
                _particlesDrawer.InitParticlesBuffer(ComputeVertexBuffer, _sizeBuffer, 0.1f);
            else
                Debug.LogWarning(
                    "If you want to visualize the change please filled the _particlesDrawer field in InteropHandlerSample monobehavior.");
        }

        /// <summary>
        ///     Create a compute buffer of SampleStructInterop of size _sizeBuffer
        /// </summary>
        private void CreateStructBuffer()
        {
            var stride = Marshal.SizeOf(typeof(SampleStructInterop));
            //allocate memory for compute buffer
            ComputeStructBuffer = new ComputeBuffer(_sizeBuffer, stride);
        }

        /// <summary>
        ///     Create the texture and the buffer. Construct action from them. Register these action in InteropUnityCUDA and
        ///     call start function on it
        /// </summary>
        public override void InitializeActions()
        {
            base.InitializeActions();

            InitSampleTexture();
            InitSampleTextureArray();
            InitSampleVertexBuffer();
            InitSampleStructBuffer();
        }

        private void InitSampleTexture()
        {
            CreateTexture();
            var actionUnitySampleTexture = new ActionUnitySampleTexture(Texture);
            RegisterActionUnity(actionUnitySampleTexture, _ActionTextureName);
            CallFunctionStartInAction(_ActionTextureName);
        }

        private void InitSampleTextureArray()
        {
            CreateTextureArray();
            var actionUnitySampleTextureArray = new ActionUnitySampleTextureArray(TextureArray);
            RegisterActionUnity(actionUnitySampleTextureArray, _ActionTextureArrayName);
            CallFunctionStartInAction(_ActionTextureArrayName);
        }

        private void InitSampleVertexBuffer()
        {
            CreateBuffer();
            var actionUnitySampleVertexBuffer = new ActionUnitySampleVertexBuffer(ComputeVertexBuffer, _sizeBuffer);
            RegisterActionUnity(actionUnitySampleVertexBuffer, _ActionVertexBufferName);
            CallFunctionStartInAction(_ActionVertexBufferName);
        }

        private void InitSampleStructBuffer()
        {
            CreateStructBuffer();
            var actionUnitySampleStructBuffer = new ActionUnitySampleStructBuffer(ComputeStructBuffer, _sizeBuffer);
            RegisterActionUnity(actionUnitySampleStructBuffer, _ActionStructBufferName);
            CallFunctionStartInAction(_ActionStructBufferName);
        }

        /// <summary>
        ///     call update function of the two registered actions
        /// </summary>
        public override void UpdateActions()
        {
            base.UpdateActions();
            CallFunctionUpdateInAction(_ActionTextureName);
            CallFunctionUpdateInAction(_ActionTextureArrayName);
            CallFunctionUpdateInAction(_ActionVertexBufferName);
            CallFunctionUpdateInAction(_ActionStructBufferName);

            Graphics.CopyTexture(TextureArray, 0, _textureForDisplay0, 0);
            Graphics.CopyTexture(TextureArray, 1, _textureForDisplay1, 0);
        }

        /// <summary>
        ///     call onDestroy function of the two registered actions
        /// </summary>
        public override void OnDestroyActions()
        {
            base.OnDestroyActions();
            CallFunctionOnDestroyInAction(_ActionTextureName);
            CallFunctionOnDestroyInAction(_ActionTextureArrayName);
            CallFunctionOnDestroyInAction(_ActionVertexBufferName);
            CallFunctionOnDestroyInAction(_ActionStructBufferName);
        }

        public struct SampleStructInterop
        {
            public float x;
            public int n;
        }
    }
}