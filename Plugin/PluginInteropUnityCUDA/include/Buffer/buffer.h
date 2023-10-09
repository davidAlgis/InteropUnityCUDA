#pragma once
#include "IUnityGraphics.h"
#include "cuda_include.h"
#include "log.h"
#include "texture.h"

/**
 * @class      Buffer
 *
 * @brief      This class give you a buffer for Unity/CUDA interoperability.
 * It's a base class for each graphics API that is implemented.
 *
 * @see \ref Buffer_D3D11
 *
 * @example action_sample_vertex_buffer.cpp shows an example of use of this
 * class with float4.
 */
class Buffer
{
    public:
    /**
     * @brief      Constructor of buffer
     *
     * @param      bufferHandle  A pointer of computeBuffer with float4 that has
     * been generated with Unity (see function GetNativeBufferPtr
     * https://docs.unity3d.com/ScriptReference/ComputeBuffer.GetNativeBufferPtr.html)
     * @param[in]  size          the size of the computeBuffer
     *
     * @example action_sample_vertex_buffer.cpp shows an example of use of this
     * class with float4.
     */
    UNITY_INTERFACE_EXPORT Buffer(void *bufferHandle, int size);

    /**
     * @brief      Register the buffer in CUDA, this has to be override because
     *             it depends on the graphics api.
     *
     * @return     0 if it works, otherwise something else.
     *
     * @example action_sample_vertex_buffer.cpp shows an example of use of this
     * class with float4.
     */
    UNITY_INTERFACE_EXPORT virtual int registerBufferInCUDA() = 0;

    /**
     * @brief      Unregister the buffer in CUDA, this has to be override
     * because it depends on the graphics api
     *
     * @return    0 if it works, otherwise something else.
     *
     * @example action_sample_vertex_buffer.cpp shows an example of use of this
     * class with float4.
     */
    UNITY_INTERFACE_EXPORT virtual int unregisterBufferInCUDA() = 0;

    /**
     * @brief      Map resources to CUDA return an array of T* defined on device
     * memory and which can be edited in CUDA.
     *
     * @param      bufferPtr  The pointer of the buffer.
     *
     * @tparam     T          The type of the buffer
     *
     * @return     0 if it works, otherwise something else.
     *
     * @example action_sample_vertex_buffer.cpp shows an example of use of this
     * class with float4.
     */
    template <typename T> UNITY_INTERFACE_EXPORT int mapResources(T **bufferPtr)
    {
        // map resource
        CUDA_CHECK_RETURN(
            cudaGraphicsMapResources(1, &_graphicsResource, nullptr));
        // pointer toward an array of float4 on device memory : the compute
        // buffer number of bytes that has been read
        size_t numBytes;
        // map the resources on a float4 array that can be modify on device
        CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
            (void **)bufferPtr, &numBytes, _graphicsResource));
        return SUCCESS_INTEROP_CODE;
    }

    /**
     * @brief      Unmap resources from CUDA This function will wait for all
     * previous GPU activity to complete
     *
     * @return     0 if it works, otherwise something else.
     *
     * @example action_sample_vertex_buffer.cpp shows an example of use of this
     * class with float4.
     */
    UNITY_INTERFACE_EXPORT int unmapResources();

    /**
     * @brief      Get the default dimension block \f$(8,1,1)\f$
     * @return     The dimension of the block.
     */
    UNITY_INTERFACE_EXPORT dim3 getDimBlock() const;

    /**
     * @brief      Get the default dimension grid \f$((sizeBuffer + 7)/8,1,1)\f$
     * @return     The dimension of the grid.
     */
    UNITY_INTERFACE_EXPORT dim3 getDimGrid() const;

    /**
     * @brief      Get the size of the buffer
     * @return     The size of the buffer.
     */
    UNITY_INTERFACE_EXPORT int getSize() const;

    protected:
    /**
     * @brief     Pointer to the buffer created in Unity
     */
    void *_bufferHandle;

    /**
     * @brief     Size of the buffer \p _bufferHandle
     */
    int _size;

    /**
     * @brief     Resource that can be used to retrieve buffer for CUDA
     */
    cudaGraphicsResource *_graphicsResource;

    private:
    /**
     * @brief     Default dimension block \f$(8,1,1)\f$
     */
    dim3 _dimBlock;

    /**
     * @brief     Default dimension grid \f$((sizeBuffer + 7)/8,1,1)\f$
     */
    dim3 _dimGrid;
};
