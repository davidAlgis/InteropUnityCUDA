#pragma once
#include "cuda_runtime.h"

// use this macro if you want to check cuda function
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        char buffer[2048];
        sprintf_s(buffer, "Cuda error: %i %s %s %d\n", code, cudaGetErrorString(code), file, line);
        std::string strError(buffer);
        Log::log().debugLogError(buffer);
    }
}


 /// <summary>
/// Get thedim grid to use for a dispatch, from a multiple of
/// dim block that are used by the kernel, and the number of
/// calculation that has to be done.
/// </summary>
/// <param name="dimBlock">Number of threads per block
/// </param> 
/// <param name="numCalculation">Number of calculation
/// to do on kernel (eg. if we make calculation on a 1024x1024 texture, and
/// we only want to compute a value on the first 528x528 pixels , then
/// numCalculation = 528,528,1)
/// </param> 
/// <param name="getUp">If true will get the
/// upper multiple of dimBlock, else will get the lower multiple. By
/// default its true.
/// </param> 
/// <param name="mustDoAllCalculation">if true
/// imply that dimBlock must be multiple of numCalculation
/// </param>
/// <returns>The dim of grid to use in dispatch</returns>
inline dim3 calculateDimGrid(dim3 dimBlock, dim3 numCalculation, bool getUp = true,
                     bool mustDoAllCalculation = false)
{
    int addFactor = getUp ? 1 : 0;
    float invDimBlockX = 1.0f / dimBlock.x;
    float invDimBlockY = 1.0f / dimBlock.y;
    float invDimBlockZ = 1.0f / dimBlock.z;

    if (mustDoAllCalculation)
    {
        if (numCalculation.x % dimBlock.x != 0 ||
            numCalculation.y % dimBlock.y != 0 ||
            numCalculation.z % dimBlock.z != 0)
        {
            Log::log().debugLogError(
                "Number of threads per block (" + std::to_string(dimBlock.x) +
                ", " + std::to_string(dimBlock.y) + ", " +
                std::to_string(dimBlock.z) +
                ")"
                " is not a multiple of (" +
                std::to_string(numCalculation.x) + ", " +
                std::to_string(numCalculation.y) + ", " +
                std::to_string(numCalculation.z) +
                ")"
                ", therefore the compute shader will not compute on all data.");
        }
    }

    unsigned int multipleDimBlockX =
        dimBlock.x * ((int)(numCalculation.x * invDimBlockX) + addFactor);
    unsigned int dimGridX = multipleDimBlockX / dimBlock.x;

    unsigned int multipleDimBlockY =
        dimBlock.y * ((int)(numCalculation.y * invDimBlockY) + addFactor);
    unsigned int dimGridY = multipleDimBlockY / dimBlock.y;

    unsigned int multipleDimBlockZ =
        dimBlock.z * ((int)(numCalculation.z * invDimBlockZ) + addFactor);
    unsigned int dimGridZ = multipleDimBlockZ / dimBlock.z;

    if (dimGridX < 1 || dimGridY < 1 || dimGridZ <1)
    {
        Log::log().debugLogError(
            "Threads group size " + std::to_string(dimGridX) +
            std::to_string(dimGridY) + std::to_string(dimGridZ) +
            " must be above zero.");
    }

    return dim3{dimGridX, dimGridY, dimGridZ};
}