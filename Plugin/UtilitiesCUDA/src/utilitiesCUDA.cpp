#include "utilitiesCUDA.h"

UTILITIES_CUDA_API void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        char buffer[2048];
        sprintf_s(buffer, "Cuda error: %i %s %s %d\n", code, cudaGetErrorString(code), file, line);
        std::string strError(buffer);
        Log::log().debugLogError(buffer);
    }
}
