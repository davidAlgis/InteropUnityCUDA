using UnityEngine;
using UnityEngine.Rendering;

namespace ArcBlanc.Utilities
{
    public class GPUUtilities 
    {

        /// <summary>
        /// Get the number of thread group to use for a dispatch, from a multiple of number of threads that are
        /// used by the kernel, and the number of calculation that has to be done.
        /// </summary>
        /// <param name="numThreads">Number of threads used by kernel you can get them using the method
        /// <c>GetKernelThreadGroupSizes()</c> in compute shader class </param>
        /// <param name="numCalculation">Number of calculation to do on kernel (eg. if we make calculation on a
        /// 1024x1024 texture, and we only want to compute a value on the first 528x528 pixels
        /// , then numCalculation = 528)</param>
        /// <param name="getUp">If true will get the upper multiple of numThreads, else will get the lower multiple.
        /// By default its true.</param>
        /// <param name="mustDoAllCalculation">if true imply that numThreads must be multiple of numCalculation</param>
        /// <returns>The number of threads groups to use in dispatch</returns>
        public static int GetThreadsGroups(uint numThreads, int numCalculation, bool getUp = true, bool mustDoAllCalculation = false)
        {
            int numThreadsInt = (int)numThreads;
            int addFactor = getUp ? 1 : 0;
            float invNumThreads = 1f / numThreadsInt;
            
            if (mustDoAllCalculation)
            {
                if (numCalculation % numThreadsInt != 0)
                {
                    Debug.LogError("Number of threads " + numThreadsInt + " is not a multiple of " + numCalculation 
                    + ", therefore the compute shader will not compute on all data.");
                }
            }
            
            
            int multipleNumThreads = numThreadsInt * ((int)(numCalculation * invNumThreads) + addFactor);
            int threadsGroups = multipleNumThreads / numThreadsInt;
            if (threadsGroups < 1)
            {
                Debug.LogError("Threads group size " + threadsGroups 
                                                     + " must be above zero. It was calculate with "
                                                     + "numThreads = " + numThreadsInt
                                                     + ", numCalculation = " + numCalculation
                                                     + ", addFactor = " + addFactor);
            }
            return multipleNumThreads/numThreadsInt;
        }

        /// <summary>
        /// Create a render texture that will be tex2D if <paramref name="volumeDepth"/> is equal to zero, else it'll be a tex2dArray
        /// TODO: make all array creation using this function (eg: the one in oceanphysics and wavecascade
        /// </summary>
        /// <param name="nbrPixelsPerRaw">width and height of the texture</param>
        /// <param name="volumeDepth">volume depth of texture</param>
        /// <param name="textureFormat">Format of the texture, by default its a single float</param>
        /// <param name="initAt0">Initialize the texture with 0 value, is less efficient</param>
        /// <returns>The new texture array</returns>
        public static RenderTexture CreateRenderTexture(
            int nbrPixelsPerRaw
            , int volumeDepth
            , RenderTextureFormat textureFormat = RenderTextureFormat.RFloat
            , bool initAt0 = false)
        {
            TextureDimension textureDimension = volumeDepth == 0 ? TextureDimension.Tex2D : TextureDimension.Tex2DArray;
            
            var rt = new RenderTexture(nbrPixelsPerRaw, nbrPixelsPerRaw, 0
                    , textureFormat, RenderTextureReadWrite.Linear)
            {
                useMipMap = false,
                autoGenerateMips = false,
                anisoLevel = 6,
                filterMode = FilterMode.Trilinear,
                wrapMode = TextureWrapMode.Clamp,
                enableRandomWrite = true,
                volumeDepth = volumeDepth,
                dimension = textureDimension,
            };

            if (initAt0)
            {
                var clearer = new TextureClearer(rt);
                clearer.ClearRenderTexture(rt);
            }
            
            return rt;
        }
    }
    
    /// <summary>
    /// This class clear a given render texture
    /// with 1,2 or 4 channels of float. RFloat, RGFloat or ARGBFloat
    /// </summary>
    public class TextureClearer
    {
        #region MEMBERS
        /// <summary>
        /// Contains the id's of the properties and the kernel of the TextureClearer compute shader. 
        /// </summary>
        private static class TextureClearerShaders
        {
            
            //Id of the properties in compute shaders
            public static readonly int Rt1Prop = Shader.PropertyToID("rt1");
            public static readonly int Rt2Prop = Shader.PropertyToID("rt2");
            public static readonly int Rt4Prop = Shader.PropertyToID("rt4");
            public static readonly int RtArray1Prop = Shader.PropertyToID("rtArray1");
            public static readonly int RtArray2Prop = Shader.PropertyToID("rtArray2");
            public static readonly int RtArray4Prop = Shader.PropertyToID("rtArray4");
            
            
            //Path of the different compute shader
            public static readonly string TextureClearerShaderPath = "ComputeShaders/PhysicsModel/TextureClearer";
            
            
            //Name of the kernel to use        
            public static readonly string TextureArrayClearer1KernelName = "TextureArrayClearer1";
            public static readonly string TextureClearer1KernelName = "TextureClearer1";
            public static readonly string TextureArrayClearer2KernelName = "TextureArrayClearer2";
            public static readonly string TextureClearer2KernelName = "TextureClearer2";
            public static readonly string TextureArrayClearer4KernelName = "TextureArrayClearer4";
            public static readonly string TextureClearer4KernelName = "TextureClearer4";

        }

        
        private readonly ComputeShader _textureClearerShader;
        private readonly int _textureClearerKernel;
        private readonly int _textureId;
        private readonly int _threadsGroupsX;
        private readonly int _threadsGroupsY;
        private readonly int _threadsGroupsZ;
        private readonly bool _isArray;
        #endregion


        public TextureClearer(RenderTexture rt) : this(NbrChannels(rt.format), rt.width, rt.volumeDepth){}

        
        /// <summary>
        ///
        /// </summary>
        public TextureClearer(int nbrChannels, int res, int volumeDepth = 0)
        {
            _textureClearerShader = Resources.Load<ComputeShader>(TextureClearerShaders.TextureClearerShaderPath);
            _isArray = volumeDepth != 0;
            string nameKernel = "";
            switch (nbrChannels)
            {
                case 1:
                    if (_isArray)
                    {
                        nameKernel = TextureClearerShaders.TextureArrayClearer1KernelName;
                        _textureId = TextureClearerShaders.RtArray1Prop;
                    }
                    else
                    {
                        nameKernel = TextureClearerShaders.TextureClearer1KernelName;
                        _textureId = TextureClearerShaders.Rt1Prop;
                    }
                    break;
                case 2:
                    if (_isArray)
                    {
                        nameKernel = TextureClearerShaders.TextureArrayClearer2KernelName;
                        _textureId = TextureClearerShaders.RtArray2Prop;
                    }
                    else
                    {
                        nameKernel = TextureClearerShaders.TextureClearer2KernelName;
                        _textureId = TextureClearerShaders.Rt2Prop;
                    }
                    break;
                case 4:
                    if (_isArray)
                    {
                        nameKernel = TextureClearerShaders.TextureArrayClearer4KernelName;
                        _textureId = TextureClearerShaders.RtArray4Prop;
                    }
                    else
                    {
                        nameKernel = TextureClearerShaders.TextureClearer4KernelName;
                        _textureId = TextureClearerShaders.Rt4Prop;
                    }
                    break;
                default:
                    Debug.LogError("Stride not supported, use float with 1 to 4 channels");
                    return;
            }
            
            _textureClearerKernel = _textureClearerShader.FindKernel(nameKernel);
            
            
            _textureClearerShader.GetKernelThreadGroupSizes(_textureClearerKernel
                , out uint numThreadsDimX
                , out uint numThreadsDimY
                , out uint numThreadsDimZ);
            
            _threadsGroupsX = GPUUtilities.GetThreadsGroups(numThreadsDimX
                , res
                , false
                , true);
            _threadsGroupsY = GPUUtilities.GetThreadsGroups(numThreadsDimY
                , res
                , false
                , true);
            if(_isArray)
            {
                _threadsGroupsZ = GPUUtilities.GetThreadsGroups(numThreadsDimZ
                    , volumeDepth
                    , false
                    , true);
            }
            else
                _threadsGroupsZ = 1;
            
            
        }
        
        
        public void ClearRenderTexture(RenderTexture rt)
        {
            
            _textureClearerShader.SetTexture(_textureClearerKernel, _textureId, rt);
            _textureClearerShader.Dispatch(_textureClearerKernel, _threadsGroupsX, _threadsGroupsY, _threadsGroupsZ);
        }
        
        
        private static int NbrChannels(RenderTextureFormat rtFormat)
        {
            int nbrChannels = 0;
            switch (rtFormat)
            {
                case RenderTextureFormat.RFloat:
                    nbrChannels = 1;
                    break;
                case RenderTextureFormat.RGFloat:
                    nbrChannels = 2;
                    break;
                case RenderTextureFormat.ARGBFloat:
                    nbrChannels = 4;
                    break;
                default:
                    Debug.LogError(rtFormat + " - Unsupported texture format ! ");
                    return -1;
            }

            return nbrChannels;
        }
    }
    
}