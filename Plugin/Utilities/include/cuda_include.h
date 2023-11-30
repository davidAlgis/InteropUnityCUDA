#pragma once
#include "cuda_runtime.h"
#include "log.h"
#include <stdexcept>

// use this macro if you want to check cuda function

/**
 * @brief      Check if a cuda function has succeed and log the cuda error if
 * it's not the case
 *
 * @param      ans   A function that return a cudaError_t
 *
 * @return     a cast into int of the return code (cudaError_t)
 * 0 if it's a success and >0 if it's an error
 */
#define CUDA_CHECK(ans) cudaAssert((ans), __FILE__, __LINE__)

/**
 * @brief      Check if a cuda function has succeed. If it doesn't log the cuda
 * error  and return the error code
 *
 * @param      ans   A function that return a cudaError_t
 *
 */
#define CUDA_CHECK_RETURN(ans)                                                 \
    {                                                                          \
        int ret = cudaAssert((ans), __FILE__, __LINE__);                       \
        if (ret != 0)                                                          \
            return ret;                                                        \
    }

/**
 * @brief      Check if a cuda function has succeed. If it doesn't log the cuda
 * error and a msg and return;
 *
 * @param      ans   The return code of the cuda function
 * @param      msg   A string to log
 *
 */
#define CUDA_CHECK_RETURN_VOID(ans, msg)                                       \
    {                                                                          \
        int ret = cudaAssert((ans), __FILE__, __LINE__);                       \
        if (ret != 0)                                                          \
        {                                                                      \
            Log::log().debugLogError(msg);                                     \
            return;                                                            \
        }                                                                      \
    }

/**
 * @brief      Check if a cuda function has succeed. If it doesn't log the cuda
 * error and a msg and return false.
 *
 * @param      ans   The return code of the cuda function
 *
 */
#define CUDA_CHECK_RETURN_BOOL(ans)                                            \
    {                                                                          \
        int ret = cudaAssert((ans), __FILE__, __LINE__);                       \
        if (ret != 0)                                                          \
        {                                                                      \
            return false;                                                      \
        }                                                                      \
    }

/**
 * @brief      Check if a cuda function has succeed. If it doesn't log the
 * cuda error and throw a runtime error
 *
 * @param      ans   A function that return a cudaError_t
 */
#define CUDA_CHECK_THROW(ans)                                                  \
    {                                                                          \
        int ret = cudaAssert((ans), __FILE__, __LINE__);                       \
        if (ret != 0)                                                          \
            throw std::runtime_error(cudaGetErrorString(ans));                 \
    }
/**
 * @brief      Check if a cufft function has succeed. If it doesn't log the
 * cufft error
 *
 * @param      ans   A function of cufft library that return an int
 *
 * @return     the return code of cufft
 * 0 if it's a success and >0 if it's an error
 */
#define CUFFT_CHECK(ans) cufftAssert((int)(ans), __FILE__, __LINE__)

/**
 * @brief      Check if a cufft function has succeed. If it doesn't log the
 * cufft error and return the error code
 *
 * @param      ans   A function of cufft library that return an int
 *
 */
#define CUFFT_CHECK_RETURN(ans)                                                \
    {                                                                          \
        int ret = cufftAssert((int)(ans), __FILE__, __LINE__);                 \
        if (ret != 0)                                                          \
            return ret;                                                        \
    }

/**
 * @brief      Log an error associated to cuda library if there has been an
 * error during a cuda function
 *
 * @param[in]  code  A return code of a function of cuda library
 * @param[in]  file  The file associated to the function call
 * @param[in]  line  The line associated to the function call
 *
 * @return     the return code
 */
inline int cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        char buffer[2048];
        sprintf_s(buffer, "Cuda error: %i %s %s %d\n", code,
                  cudaGetErrorString(code), file, line);
        std::string strError(buffer);
        Log::log().debugLogError(buffer);
    }
    return (int)code;
}

/**
 * @brief      Log an error associated to cufft library if there has been an
 * error during a cufft function
 *
 * @param[in]  cufftResult  A return code of a function of cufft library
 * @param[in]  file         The file associated to the function call
 * @param[in]  line         The line associated to the function call
 *
 * @return     the return code
 */
inline int cufftAssert(int cufftResult, const char *file, int line)
{

    if (cufftResult != 0)
    {
        std::string cufftInterpret;
        switch (cufftResult)
        {
        case (0):
            cufftInterpret = "The cuFFT operation was successful";
            break;
        case (1):
            cufftInterpret = "cuFFT was passed an invalid plan handle";
            break;
        case (2):
            cufftInterpret = "cuFFT failed to allocate GPU or CPU memory";
            break;
        case (3):
            cufftInterpret = "No longer used";
            break;
        case (4):
            cufftInterpret = "User specified an invalid pointer or parameter";
            break;
        case (5):
            cufftInterpret = "Driver or internal cuFFT library error";
            break;
        case (6):
            cufftInterpret = "Failed to execute an FFT on the GPU";
            break;
        case (7):
            cufftInterpret = "The cuFFT library failed to initialize";
            break;
        case (8):
            cufftInterpret = "User specified an invalid transform size";
            break;
        case (9):
            cufftInterpret = "No longer used";
            break;
        case (10):
            cufftInterpret = "Missing parameters in call";
            break;
        case (11):
            cufftInterpret =
                "Execution of a plan was on different GPU than plan creation";
            break;
        case (12):
            cufftInterpret = "Internal plan database error";
            break;
        case (13):
            cufftInterpret =
                "No workspace has been provided prior to plan execution";
            break;
        case (14):
            cufftInterpret = "Function does not implement functionality for "
                             "parameters given.";
            break;
        case (15):
            cufftInterpret = "Used in previous versions.";
            break;
        case (16):
            cufftInterpret = "Operation is not supported for parameters given.";
            break;
        default:
            cufftInterpret = "Unknown error.";
            break;
        }
        char buffer[2048];
        sprintf_s(buffer, "Cufft error: %i %s %s %d\n", cufftResult,
                  cufftInterpret.c_str(), file, line);
        std::string strError(buffer);
        Log::log().debugLogError(buffer);
    }
    return cufftResult;
}

/**
 * @brief      Get the dim grid to use for a dispatch, from a multiple of
 * dim block that are used by the kernel, and the number of
 * calculation that has to be done.
 *
 * @param[in]  dimBlock              Number of threads per block
 * @param[in]  numCalculation        Number of calculation
 * to do on kernel (eg. if we make calculation on a 1024x1024 texture, and
 * we only want to compute a value on the first 528x528 pixels , then
 * numCalculation = 528,528,1)
 * @param[in]  getUp                 If true will get the upper multiple of
 * dimBlock, else will get the lower multiple. By default its true.
 * @param[in]  mustDoAllCalculation  Imply that dimBlock must
 * be multiple of numCalculation
 *
 * @return     The dim of grid to use in dispatch
 */
inline dim3 calculateDimGrid(dim3 dimBlock, dim3 numCalculation,
                             bool getUp = true,
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
                ("Number of threads per block (" + std::to_string(dimBlock.x) +
                 ", " + std::to_string(dimBlock.y) + ", " +
                 std::to_string(dimBlock.z) +
                 ")"
                 " is not a multiple of (" +
                 std::to_string(numCalculation.x) + ", " +
                 std::to_string(numCalculation.y) + ", " +
                 std::to_string(numCalculation.z) +
                 ")"
                 ", therefore the compute shader will not compute on all data.")
                    .c_str());
        }
    }

    unsigned int multipleDimBlockX =
        dimBlock.x * ((int)(numCalculation.x * invDimBlockX) + addFactor);
    // unsigned int multipleDimBlockX =
    //     dimBlock.x * (numCalculation.x / dimBlock.x) + addFactor);
    //  TODO remove dimBlock.x above and bellow
    unsigned int dimGridX = multipleDimBlockX / dimBlock.x;

    unsigned int multipleDimBlockY =
        dimBlock.y * ((int)(numCalculation.y * invDimBlockY) + addFactor);
    unsigned int dimGridY = multipleDimBlockY / dimBlock.y;

    unsigned int multipleDimBlockZ =
        dimBlock.z * ((int)(numCalculation.z * invDimBlockZ) + addFactor);
    unsigned int dimGridZ = multipleDimBlockZ / dimBlock.z;

    if (dimGridX < 1 || dimGridY < 1 || dimGridZ < 1)
    {
        Log::log().debugLogError(
            ("Threads group size " + std::to_string(dimGridX) +
             std::to_string(dimGridY) + std::to_string(dimGridZ) +
             " must be above zero.")
                .c_str());
    }

    return dim3{dimGridX, dimGridY, dimGridZ};
}

/**
 * @brief      Check if dim grid and dim block are of a correct size
 *
 * @param[in]  dimGrid   The dimension of the grid to check
 * @param[in]  dimBlock  The dimension of the block to check
 * @param[in]  device    The device number to check on
 *
 * @return     true if grid and block size are of the correct size, else false
 */
UTILITIES_DLL_EXPORT inline bool checkCUDAConfiguration(const uint32_t dimGrid,
                                                        const uint32_t dimBlock,
                                                        int device = 0)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if (dimGrid > static_cast<uint32_t>(deviceProp.maxGridSize[0]) ||
        dimBlock > static_cast<uint32_t>(maxThreadsPerBlock))
    {
        Log::log().debugLogError(
            ("Invalid grid or block size configuration. Maximum grid size: " +
             std::to_string(deviceProp.maxGridSize[0]) +
             "Maximum threads per block: " + std::to_string(maxThreadsPerBlock))
                .c_str());
        return false;
    }

    return true;
}