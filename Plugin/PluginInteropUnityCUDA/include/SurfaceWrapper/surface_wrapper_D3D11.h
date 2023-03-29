#pragma once
#include "surface_wrapper.h"

#if SUPPORT_D3D11

template <class T> class Surface_D3D11 : public SurfaceWrapper<T>
{
    public:
    __device__ Surface_D3D11(cudaGraphicsResource *_pGraphicsResource)
    {
        
    }

    __device__ ~Surface_D3D11()
    {
    }

    __device__ virtual void surface3DWrite(T data, int x, int y, int z) override
    {
        // surf2Dwrite(data, _surfObjectsArray[z], sizeof(T) * x, y);
    }

    __device__ virtual T surface3DRead(int x, int y, int z) override
    {
        return make_float4(0,0,0,1);
        // return surf2Dread(_surfObjectsArray[z], sizeof(T) * x, y);
    }

    private:
    cudaSurfaceObject_t *_surfObjectsArray;
};

#endif