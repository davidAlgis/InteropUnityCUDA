#pragma once
#include "surface_wrapper.h"

#if SUPPORT_OPENGL_UNIFIED

template <class T> class Surface_OpenGLCoreES : public SurfaceWrapper<T>
{
    public:
    __device__ Surface_OpenGLCoreES(cudaGraphicsResource *_pGraphicsResource)
    {
    }

    __device__ ~Surface_OpenGLCoreES()
    {
    }

    __device__ virtual void surface3DWrite(T data, int x, int y, int z) override
    {
        // surf3Dwrite(data, _surfObject, sizeof(T) * x, y, z);
    }

    __device__ virtual T surface3DRead(int x, int y, int z) override
    {
        return make_float4(0,0,0,1);
        // return surf3Dread(_surfObject, sizeof(T) * x, y, z);
    }

    private:
    cudaSurfaceObject_t _surfObject;
};

#endif