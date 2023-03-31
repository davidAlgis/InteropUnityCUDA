#include "surface_wrapper_device_allocator.cuh"

/**
 * Allocate the surface wrapper on device memory for DX11 api.
 * This is necessary to use surfaceWrapper because it's an abstract class
 * with virtual device method and therefore its allocation has to be done
 * on device memory.
 * see. https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
 * @param surfaceWrapper     the surface wrapper pointer to f
 * @param _pGraphicsResource the graphics ressources needed for surface wrapper
 * @param textureDepth       the texture depth
 */
__global__ void createSurfaceWrapper_DX11(
    SurfaceWrapper<float4> **surfaceWrapper, cudaGraphicsResource *_pGraphicsResource, int textureDepth)
{
    (*surfaceWrapper) = new Surface_D3D11<float4>(_pGraphicsResource, textureDepth);
}

/**
 * Allocate the surface wrapper on device memory for OpenGL api.
 * This is necessary to use surfaceWrapper because it's an abstract class
 * with virtual device method and therefore its allocation has to be done
 * on device memory.
 * see. https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
 * @param surfaceWrapper     the surface wrapper pointer to f
 * @param _pGraphicsResource the graphics ressources needed for surface wrapper
 */
__global__ void createSurfaceWrapper_OpenGLCoreES(
    SurfaceWrapper<float4> **surfaceWrapper, cudaGraphicsResource *_pGraphicsResource)
{
    (*surfaceWrapper) = new Surface_OpenGLCoreES<float4>(_pGraphicsResource);
}


/**
 * Free in device memory the surface wrapper. 
 * you MUST call this function after calling a createSurfaceWrapper on any graphics API
 * @param surfaceWrapper the surface wrapper pointer to free 
 */
__global__ void deleteSurfaceWrapper(SurfaceWrapper<float4> **surfaceWrapper)
{
    delete *surfaceWrapper;
}


void kernelCallerCreateSurfaceWrapper_DX11(
    SurfaceWrapper<float4> **surfaceWrapper, cudaGraphicsResource *_pGraphicsResource, int textureDepth)
{
    createSurfaceWrapper_DX11<<<1, 1>>>(surfaceWrapper, _pGraphicsResource, textureDepth);
}

void kernelCallerCreateSurfaceWrapper_OpenGLCoreES(
    SurfaceWrapper<float4> **surfaceWrapper, cudaGraphicsResource *_pGraphicsResource)
{
    createSurfaceWrapper_OpenGLCoreES<<<1, 1>>>(surfaceWrapper, _pGraphicsResource);
}


void kernelCallerDeleteSurfaceWrapper(SurfaceWrapper<float4> **surfaceWrapper)
{
    deleteSurfaceWrapper<<<1, 1>>>(surfaceWrapper);
}