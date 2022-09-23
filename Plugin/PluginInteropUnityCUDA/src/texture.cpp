#include "texture.h"


void kernelCallerWriteTexture(const dim3 dimGrid,const dim3 dimBlock, cudaSurfaceObject_t inputSurfaceObj, const float t, const int width, const int height);

void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        char buffer[2048];
        sprintf_s(buffer, "Cuda error: %i %s %s %d\n", code, cudaGetErrorString(code), file, line);
        std::string strError(buffer);
        Log::log().debugLogError(buffer);
    }
}




Texture::Texture(void* textureHandle, int textureWidth, int textureHeight)
{
    _textureHandle = textureHandle;
    _textureWidth = textureWidth;
    _textureHeight = textureHeight;
    _dimBlock = { 8, 8, 1 };
    _dimGrid = { ((textureWidth + _dimBlock.x - 1) / _dimBlock.x,
        (textureHeight + _dimBlock.y - 1) / _dimBlock.y), 1};
    
}


cudaSurfaceObject_t Texture::mapTextureToSurfaceObject()
{
    cudaArray* arrayPtr;
    cudaGraphicsMapResources(1, &_pGraphicsResource);
    //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&arrayPtr, _pGraphicsResource, 0, 0));

    // Wrap the cudaArray in a surface object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arrayPtr;
    cudaSurfaceObject_t inputSurfObj = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&inputSurfObj, &resDesc));
    return inputSurfObj;
}

void Texture::writeTexture(cudaSurfaceObject_t& inputSurfObj)
{
    kernelCallerWriteTexture(_dimGrid, _dimBlock, inputSurfObj, 0, _textureWidth, _textureHeight);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Texture::unMapTextureToSurfaceObject(cudaSurfaceObject_t& inputSurfObj)
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
    CUDA_CHECK(cudaDestroySurfaceObject(inputSurfObj));
    CUDA_CHECK(cudaGetLastError());
}



Texture* createTextureAPI(void* textureHandle, int textureWidth, int textureHeight, UnityGfxRenderer apiType)
{
//#	if SUPPORT_D3D11
//	if (apiType == kUnityGfxRendererD3D11)
//	{
//		extern Texture* CreateRenderAPI_D3D11();
//		return CreateRenderAPI_D3D11();
//	}
//#	endif // if SUPPORT_D3D11
//
//#	if SUPPORT_D3D12
//	if (apiType == kUnityGfxRendererD3D12)
//	{
//		extern RenderAPI* CreateRenderAPI_D3D12();
//		return CreateRenderAPI_D3D12();
//	}
//#	endif // if SUPPORT_D3D12


#	if SUPPORT_OPENGL_UNIFIED
	if (apiType == kUnityGfxRendererOpenGLCore || apiType == kUnityGfxRendererOpenGLES20 || apiType == kUnityGfxRendererOpenGLES30)
	{
		extern Texture* createTextureAPI_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight);
		return createTextureAPI_OpenGLCoreES(textureHandle, textureWidth, textureHeight);
	}
#	endif // if SUPPORT_OPENGL_UNIFIED

	// Unknown or unsupported graphics API
	return NULL;
}
