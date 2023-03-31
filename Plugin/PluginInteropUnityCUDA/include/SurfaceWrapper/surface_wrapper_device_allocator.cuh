#pragma once
#include "cuda_include.h"
#include "surface_wrapper_D3D11.h"
#include "surface_wrapper_OpenGLCoreES.h"

/**
 * Call the kernel that allocate the surface wrapper on device memory for DX11 api.
 * This is necessary to use surfaceWrapper because it's an abstract class
 * with virtual device method and therefore its allocation has to be done
 * on device memory.
 * see. https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
 * @param surfaceWrapper     the surface wrapper pointer to f
 * @param _pGraphicsResource the graphics ressources needed for surface wrapper
 */
void kernelCallerCreateSurfaceWrapper_DX11(SurfaceWrapper<float4> **a, cudaGraphicsResource *_pGraphicsResource);


/**
 * Call the kernel that allocate the surface wrapper on device memory for OpenGL api.
 * This is necessary to use surfaceWrapper because it's an abstract class
 * with virtual device method and therefore its allocation has to be done
 * on device memory.
 * see. https://stackoverflow.com/questions/26812913/how-to-implement-device-side-cuda-virtual-functions
 * @param surfaceWrapper     the surface wrapper pointer to f
 * @param _pGraphicsResource the graphics ressources needed for surface wrapper
 */
void kernelCallerCreateSurfaceWrapper_OpenGLCoreES(SurfaceWrapper<float4> **a, cudaGraphicsResource *_pGraphicsResource);

/**
 * Call the kernel that free in device memory the surface wrapper. 
 * you MUST call this function after calling a createSurfaceWrapper on any graphics API
 * @param surfaceWrapper the surface wrapper pointer to free 
 */
void kernelCallerDeleteSurfaceWrapper(SurfaceWrapper<float4> **a);