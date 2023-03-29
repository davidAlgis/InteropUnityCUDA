#pragma once
#include "cuda_include.h"
#include "log.h"


template <class T> class SurfaceWrapper
{
    public:
    __device__ SurfaceWrapper(){};
    __device__ virtual ~SurfaceWrapper(){};
    __device__ virtual void surface3DWrite(T data, int x, int y, int z) = 0;
    __device__ virtual T surface3DRead(int x, int y, int z) = 0;
};
