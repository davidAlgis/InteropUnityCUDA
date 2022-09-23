#pragma once
#include "cuda_runtime.h"
#include "log.h"
#include "frameworkUtilitiesCUDA.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
UTILITIES_CUDA_API void gpuAssert(cudaError_t code, const char* file, int line);