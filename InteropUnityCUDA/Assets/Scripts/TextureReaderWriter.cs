using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace ArcBlanc.Utilities
{
    /// <summary>
    /// This class can read and write into a render texture
    /// with 1,2 or 4 channels of float. RFloat, RGFloat or ARGBFloat
    /// This class was created for DEBUGGING purposes !!!
    /// Therefore, if you need it for another purposes I advice you to
    /// change NUM_THREADS_DIM_X,Y in TextureReaderWriter.compute and
    /// NumThreadsDimX,Y bellow, to improve performance according to texture
    /// size.
    ///
    /// NOTE ON ORIENTATION :
    /// Texture is write and read with this basis :
    ///             (2nd index) 
    ///                 ↑
    ///                 |
    ///                 | 
    ///                 |    
    ///                 |-----------→ (1st index)
    /// </summary>
    public class TextureReaderWriter
    {
        #region MEMBERS
        /// <summary>
        /// Contains the id's of the properties and the kernel of the TextureReaderWriter compute shader. 
        /// </summary>
        private static class TextureReaderWriterShaders
        {
            
            //Id of the properties in compute shaders
            public static readonly int Rt1Prop = Shader.PropertyToID("rt1");
            public static readonly int Rt2Prop = Shader.PropertyToID("rt2");
            public static readonly int Rt4Prop = Shader.PropertyToID("rt4");
            
            public static readonly int ArrayIn1Prop = Shader.PropertyToID("arrayIn1");
            public static readonly int ArrayIn2Prop = Shader.PropertyToID("arrayIn2");
            public static readonly int ArrayIn4Prop = Shader.PropertyToID("arrayIn4");
            
            public static readonly int ArrayOut1Prop = Shader.PropertyToID("arrayOut1");
            public static readonly int ArrayOut2Prop = Shader.PropertyToID("arrayOut2");
            public static readonly int ArrayOut4Prop = Shader.PropertyToID("arrayOut4");
            
            public static readonly int ResProp = Shader.PropertyToID("res");
            
            
            //Path of the different compute shader
            public static readonly string TextureReaderWriterShaderPath = "ComputeShaders/TextureReaderWriter";
            
            
            //Name of the kernel to use        
            public static readonly string ArrayToRenderTexture1KernelName = "ArrayToRenderTexture1";
            public static readonly string RenderTextureToArray1KernelName = "RenderTextureToArray1";
            public static readonly string ArrayToRenderTexture2KernelName = "ArrayToRenderTexture2";
            public static readonly string RenderTextureToArray2KernelName = "RenderTextureToArray2";
            public static readonly string ArrayToRenderTexture4KernelName = "ArrayToRenderTexture4";
            public static readonly string RenderTextureToArray4KernelName = "RenderTextureToArray4";
            
            /*Number of threads used in compute shader, if these variable are changed,
             *they must be changed too in compute shader DebugTexture.compute*/
            public const int NumThreadsDimX = 1;
            public const int NumThreadsDimY = 1;
            
            
        }

        
        private readonly ComputeShader _textureReaderWriterShader;
        private readonly int _arrayToRenderTextureKernel;
        private readonly int _renderTextureToArrayKernel;
        
        private readonly ComputeBuffer _bufferIn;
        private readonly ComputeBuffer _bufferOut;
        
        private float[] _arrayIn1;
        private float[,] _arrayIn2D1;
        private float2[] _arrayIn2;
        private float2[,] _arrayIn2D2;
        private float4[] _arrayIn4;
        private float4[,] _arrayIn2D4;
        
        private float[] _arrayOut1;
        private float[,] _arrayOut2D1;
        private float2[] _arrayOut2;
        private float2[,] _arrayOut2D2;
        private float4[] _arrayOut4;
        private float4[,] _arrayOut2D4;
        
        private readonly int _nbrChannels;
        private readonly int _res;
        
        
        private readonly int _threadsGroupsX;
        private readonly int _threadsGroupsY;

        private readonly int _renderTextureId;
        private readonly int _arrayInId;
        private readonly int _arrayOutId;
        
        
        #region MEMBERS_PROPERTIES

        public int Res => _res;

        /// <summary>
        /// The value in this array will be written in render texture
        /// </summary>
        public float[] ArrayIn1
        {
            get => _arrayIn1;
            set => _arrayIn1 = value;
        }

        /// <summary>
        /// The value in this array will be written in render texture
        /// </summary>
        public float2[] ArrayIn2
        {
            get => _arrayIn2;
            set => _arrayIn2 = value;
        }

        /// <summary>
        /// The value in this array will be written in render texture
        /// </summary>
        public float4[] ArrayIn4
        {
            get => _arrayIn4;
            set => _arrayIn4 = value;
        }

        /// <summary>
        /// The value in this array will be set from the value that has been read in render texture
        /// </summary>
        public float[] ArrayOut1
        {
            get => _arrayOut1;
            set => _arrayOut1 = value;
        }

        /// <summary>
        /// The value in this array will be set from the value that has been read in render texture
        /// </summary>
        public float2[] ArrayOut2
        {
            get => _arrayOut2;
            set => _arrayOut2 = value;
        }


        /// <summary>
        /// The value in this array will be set from the value that has been read in render texture
        /// </summary>
        public float4[] ArrayOut4
        {
            get => _arrayOut4;
            set => _arrayOut4 = value;
        }

        public float[,] ArrayIn2D1
        {
            get => _arrayIn2D1;
            set => _arrayIn2D1 = value;
        }

        public float2[,] ArrayIn2D2
        {
            get => _arrayIn2D2;
            set => _arrayIn2D2 = value;
        }

        public float4[,] ArrayIn2D4
        {
            get => _arrayIn2D4;
            set => _arrayIn2D4 = value;
        }

        public float[,] ArrayOut2D1
        {
            get => _arrayOut2D1;
            set => _arrayOut2D1 = value;
        }

        public float2[,] ArrayOut2D2
        {
            get => _arrayOut2D2;
            set => _arrayOut2D2 = value;
        }
        public float4[,] ArrayOut2D4
        {
            get => _arrayOut2D4;
            set => _arrayOut2D4 = value;
        }

        #endregion
        
        #endregion
        
        /// <summary>
        /// This class can read and write into a render texture
        /// with 1,2,3 or 4 channels of float. RFloat, RGFloat, RGBFloat
        /// or ARGBFloat
        /// This class was created for DEBUGGING purposes !!!
        /// Therefore, if you need it for another purposes I advice you to
        /// change NUM_THREADS_DIM_X,Y in TextureReaderWriter.compute and
        /// NumThreadsDimX,Y bellow, to improve performance according to texture
        /// size.
        /// WARNING : DON'T FORGET TO CALL DESTROY AFTER USING THIS CLASS.
        /// This will correctly dispose the memory.
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="resTexture">Resolution of the texture</param>
        /// <param name="nbrChannels">Number of the channels in the texture</param>
        public TextureReaderWriter(int resTexture, int nbrChannels)
        {
            int stride;
            int fullRes = resTexture * resTexture;
            
            _res = resTexture;
            
            _nbrChannels = nbrChannels;
            _textureReaderWriterShader = Resources.Load<ComputeShader>(TextureReaderWriterShaders.TextureReaderWriterShaderPath);

            switch (nbrChannels)
            {
                case 1:
                    stride = Marshal.SizeOf(typeof(float));
                    _arrayIn1 = new float[fullRes];
                    _arrayOut1 = new float[fullRes];
                    _arrayIn2D1 = new float[resTexture, resTexture];
                    _arrayOut2D1 = new float[resTexture, resTexture];
                    
                    _renderTextureId = TextureReaderWriterShaders.Rt1Prop;
                    _arrayInId = TextureReaderWriterShaders.ArrayIn1Prop;
                    _arrayOutId = TextureReaderWriterShaders.ArrayOut1Prop;
                    
                    _arrayToRenderTextureKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.ArrayToRenderTexture1KernelName);
                    _renderTextureToArrayKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.RenderTextureToArray1KernelName);
                    
                    break;
                case 2:
                    stride = Marshal.SizeOf(typeof(float2));
                    _arrayIn2 = new float2[fullRes];
                    _arrayOut2 = new float2[fullRes];
                    _arrayIn2D2 = new float2[resTexture, resTexture];
                    _arrayOut2D2 = new float2[resTexture, resTexture];
                    
                    _renderTextureId = TextureReaderWriterShaders.Rt2Prop;
                    _arrayInId = TextureReaderWriterShaders.ArrayIn2Prop;
                    _arrayOutId = TextureReaderWriterShaders.ArrayOut2Prop;
                    
                    _arrayToRenderTextureKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.ArrayToRenderTexture2KernelName);
                    _renderTextureToArrayKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.RenderTextureToArray2KernelName);
                    
                    break;
                case 4:
                    stride = Marshal.SizeOf(typeof(float4)); 
                    _arrayIn4 = new float4[fullRes];
                    _arrayOut4 = new float4[fullRes];
                    _arrayIn2D4 = new float4[resTexture, resTexture];
                    _arrayOut2D4 = new float4[resTexture, resTexture];
                    
                    _renderTextureId = TextureReaderWriterShaders.Rt4Prop;
                    _arrayInId = TextureReaderWriterShaders.ArrayIn4Prop;
                    _arrayOutId = TextureReaderWriterShaders.ArrayOut4Prop;
                    
                    _arrayToRenderTextureKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.ArrayToRenderTexture4KernelName);
                    _renderTextureToArrayKernel = _textureReaderWriterShader.FindKernel(TextureReaderWriterShaders.RenderTextureToArray4KernelName);
                    
                    break;
                default:
                    Debug.LogError("Stride not supported, use float with 1 to 4 channels");
                    return;
            }
            
            
            _bufferIn = new ComputeBuffer(fullRes, stride, ComputeBufferType.Default);
            _bufferOut = new ComputeBuffer(fullRes, stride, ComputeBufferType.Default);
            
            
            if (_textureReaderWriterShader == null)
            {
                Debug.LogError("Unable to find the compute shader at path "+ TextureReaderWriterShaders.TextureReaderWriterShaderPath);
                return;
            }
            
            
            _threadsGroupsX = GPUUtilities.GetThreadsGroups(TextureReaderWriterShaders.NumThreadsDimX
            , _res
            , false);
            _threadsGroupsY = GPUUtilities.GetThreadsGroups(TextureReaderWriterShaders.NumThreadsDimY
                , _res
                , false);
        }

        #region RENDERTEXTURE -> ARRAY

        /// <summary>
        /// Write directly into the array of the class to read from the texture
        /// Caution : use the array that match the number of channels
        /// float : _arrayOut1 or _arrayOut2D1
        /// float2 : _arrayOut2 or _arrayOut2D2
        /// float3 : _arrayOut3 or _arrayOut2D3
        /// float4 : _arrayOut4 or _arrayOut2D4
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into, if it's a texture array we will only read the first texture</param>
        /// <param name="copyTo2DArray">if true will copy the memory in 2d Array too. This can impact the performances badly</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, bool copyTo2DArray = true, int indexInVolumeDepth = 0)
        {
            switch (_nbrChannels)
            {
                case 1:
                    RenderTextureReaderToArray(rt, out _arrayOut1);
                    if(copyTo2DArray)
                        MathUtilities.Array1DToArray2D(_arrayOut1, ref _arrayOut2D1);
                    break;
                case 2:
                    RenderTextureReaderToArray(rt, out _arrayOut2);
                    if(copyTo2DArray)
                        MathUtilities.Array1DToArray2D(_arrayOut2, ref _arrayOut2D2);
                    break;
                case 4:
                    RenderTextureReaderToArray(rt, out _arrayOut4);
                    if(copyTo2DArray)
                        MathUtilities.Array1DToArray2D(_arrayOut4, ref _arrayOut2D4);
                    break;
                default:
                    Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                    return;
            }
        }
        
        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">2D array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float[,] array, int indexInVolumeDepth = 0)
        {
            array = new float[_res, _res];
            RenderTextureReaderToArray(rt, out _arrayOut1, indexInVolumeDepth);
            MathUtilities.Array1DToArray2D(_arrayOut1, ref array);
        }
        
        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float[] array, int indexInVolumeDepth = 0)
        {
            if (RenderTextureToArrayDispatch(rt, 1,indexInVolumeDepth) == false)
            {
                array = null;
                return;
            }
            _bufferOut.GetData(_arrayIn1);
            array = _arrayIn1;
        }

        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">2D array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float2[,] array, int indexInVolumeDepth = 0)
        {
            array = new float2[_res, _res];
            RenderTextureReaderToArray(rt, out _arrayOut2, indexInVolumeDepth);
            MathUtilities.Array1DToArray2D(_arrayOut2, ref array);
        }
        

        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float2[] array, int indexInVolumeDepth = 0)
        {
            
            if (RenderTextureToArrayDispatch(rt, 2,indexInVolumeDepth) == false)
            {
                array = null;
                return;
            }
            _bufferOut.GetData(_arrayIn2);
            array = _arrayIn2;
        }
        
        
        
        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">2D array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float4[,] array, int indexInVolumeDepth = 0)
        {
            array = new float4[_res, _res];
            RenderTextureReaderToArray(rt, out _arrayOut4, indexInVolumeDepth);
            MathUtilities.Array1DToArray2D(_arrayOut4, ref array);
        }
        
        
        /// <summary>
        /// Read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="array">array to write from the render texture</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void RenderTextureReaderToArray(RenderTexture rt, out float4[] array, int indexInVolumeDepth = 0)
        {
            if (RenderTextureToArrayDispatch(rt, 4, indexInVolumeDepth) == false)
            {
                array = null;
                return;
            }
            _bufferOut.GetData(_arrayIn4);
            array = _arrayIn4;
        }

        
        /// <summary>
        /// Dispatch the kernel that read into the render texture and write into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="nbrChannel">Nbr the channel</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        private bool RenderTextureToArrayDispatch(RenderTexture rt, int nbrChannel, int indexInVolumeDepth )
        {
            if (_nbrChannels != nbrChannel)
            {
                Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                return false;
            }
                
            RenderTexture rtOneElement;
            
            if (rt.volumeDepth > 0)
            {
                rtOneElement = GPUUtilities.CreateRenderTexture(rt.height, 0, rt.format);
                Graphics.CopyTexture(rt, indexInVolumeDepth, rtOneElement, 0);
            }
            else
            {
                rtOneElement = rt;
            }
            
            _textureReaderWriterShader.SetInt(TextureReaderWriterShaders.ResProp, _res);
            _textureReaderWriterShader.SetTexture(_renderTextureToArrayKernel, _renderTextureId, rtOneElement);
            _textureReaderWriterShader.SetBuffer(_renderTextureToArrayKernel, _arrayOutId, _bufferOut);
            _textureReaderWriterShader.Dispatch(_renderTextureToArrayKernel, _threadsGroupsX, _threadsGroupsY, 1);
            return true;

        }
        

        #endregion

        #region ARRAY -> RENDERTEXTURE

        /// <summary>
        /// Use directly the array of the class to write into the render texture
        /// Caution : use the array that match the number of channels
        /// float : _arrayIn1 or _arrayIn2D1
        /// float2 : _arrayIn2 or _arrayIn2D2
        /// float4 : _arrayIn4 or _arrayIn2D4
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="use1DArray">if true use 1D array, else will use 2D array</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(ref RenderTexture rt, bool use1DArray, int indexInVolumeDepth = 0)
        {
            switch (_nbrChannels)
            {
                case 1:
                    if(use1DArray)
                        ArrayToRenderTextureWriter(_arrayIn1, ref rt);
                    else
                        ArrayToRenderTextureWriter(_arrayIn2D1, ref rt);
                    break;
                case 2:
                    if(use1DArray)
                        ArrayToRenderTextureWriter(_arrayIn2, ref rt);
                    else
                        ArrayToRenderTextureWriter(_arrayIn2D2, ref rt);
                    break;
                case 4:
                    if(use1DArray)
                        ArrayToRenderTextureWriter(_arrayIn4, ref rt);
                    else
                        ArrayToRenderTextureWriter(_arrayIn2D4, ref rt);
                    break;
                default:
                    Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                    return;
            }
            
        }
        
        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">2d array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float[,] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            MathUtilities.Array2DToArray1D(array, ref _arrayIn1);
            ArrayToRenderTextureWriter(_arrayIn1, ref rt, indexInVolumeDepth);
        }

        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float[] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {

            if (_nbrChannels != 1)
            {
                Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                return;
            }
            
            _bufferIn.SetData(array);
            ArrayToRenderTextureDispatch(ref rt, indexInVolumeDepth);
        }

        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">2d array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float2[,] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            MathUtilities.Array2DToArray1D(array, ref _arrayIn2);
            ArrayToRenderTextureWriter(_arrayIn2, ref rt, indexInVolumeDepth);
        }
        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float2[] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            
            if (_nbrChannels != 2)
            {
                Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                return;
            }
            _bufferIn.SetData(array);
            ArrayToRenderTextureDispatch(ref rt, indexInVolumeDepth);
        }
        
        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float3[] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            
            if (_nbrChannels != 3)
            {
                Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                return;
            }
            _bufferIn.SetData(array);
            ArrayToRenderTextureDispatch(ref rt, indexInVolumeDepth);
        }
        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION (see scheme in code):
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">2d array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float4[,] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            MathUtilities.Array2DToArray1D(array, ref _arrayIn4);
            ArrayToRenderTextureWriter(_arrayIn4, ref rt, indexInVolumeDepth);
        }
        
        /// <summary>
        /// Write into the render texture by reading into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to write into</param>
        /// <param name="array">array read</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        public void ArrayToRenderTextureWriter(float4[] array, ref RenderTexture rt, int indexInVolumeDepth = 0)
        {
            
            if (_nbrChannels != 4)
            {
                Debug.LogError("Unable to set array of float, because the texture has " + _nbrChannels);
                return;
            }
            _bufferIn.SetData(array);
            ArrayToRenderTextureDispatch(ref rt, indexInVolumeDepth);
        }

        /// <summary>
        /// Dispatch the kernel that write into the render texture and read into the array
        /// NOTE ON ORIENTATION :
        /// Texture is write and read with this basis :
        ///             (2nd index) 
        ///                 ↑
        ///                 |
        ///                 | 
        ///                 |    
        ///                 |-----------→ (1st index)
        /// </summary>
        /// <param name="rt">render texture to read into</param>
        /// <param name="indexInVolumeDepth">If the texture is a texture array it'll copy this index in the volume depth</param>
        private void ArrayToRenderTextureDispatch(ref RenderTexture rt, int indexInVolumeDepth)
        {
            RenderTexture rtOneElement;
            bool needToCopy = false;
            if (rt.volumeDepth > 0)
            {
                rtOneElement = GPUUtilities.CreateRenderTexture(rt.height, 0, rt.format);
                Graphics.CopyTexture(rt, indexInVolumeDepth, rtOneElement, 0);
                needToCopy = true;
            }
            else
            {
                rtOneElement = rt;
            }
            
            _textureReaderWriterShader.SetInt(TextureReaderWriterShaders.ResProp, _res);
            _textureReaderWriterShader.SetTexture(_arrayToRenderTextureKernel, _renderTextureId, rtOneElement);
            _textureReaderWriterShader.SetBuffer(_arrayToRenderTextureKernel, _arrayInId, _bufferIn);
            _textureReaderWriterShader.Dispatch(_arrayToRenderTextureKernel, _threadsGroupsX, _threadsGroupsY, 1);
            if(needToCopy)
                Graphics.CopyTexture(rtOneElement, 0, rt, indexInVolumeDepth);
            
        }

        #endregion

        public int Get1DIndexFrom2DIndex(int2 id)
        {
            return id.x * _res + id.y;
        }
        

        /// <summary>
        /// Dispose the buffer
        /// </summary>
        public void Destroy()
        {
            _bufferIn.Dispose();
            _bufferOut.Dispose();
        }

        
    }
}