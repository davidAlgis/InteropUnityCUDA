#pragma once
#include "cuda_include.h"
#include "surface_wrapper_D3D11.h"
#include "surface_wrapper_OpenGLCoreES.h"

__global__ void createSurfaceWrapper_DX11(
    SurfaceWrapper<float4> **a, cudaGraphicsResource *_pGraphicsResource)
{
    (*a) = new Surface_D3D11<float4>(_pGraphicsResource);
}

__global__ void createSurfaceWrapper_OpenGLCoreES(
    SurfaceWrapper<float4> **a, cudaGraphicsResource *_pGraphicsResource)
{
    (*a) = new Surface_OpenGLCoreES<float4>(_pGraphicsResource);
}

__global__ void deleteSurfaceWrapper(SurfaceWrapper<float4> **a)
{
    delete *a;
}