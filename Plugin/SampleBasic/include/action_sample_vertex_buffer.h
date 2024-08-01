#pragma once
#include "action.h"
#include "buffer.h"
#include "sample_kernels.cuh"

namespace SampleBasic
{
/**
 * @class      ActionSampleVertexBuffer
 *
 * @brief      This class shows an example of used of \ref Buffer class
 * of \p PluginInteropUnityCUDA library.
 * More precisely, it demonstrate how by given a
 * <a href="https://docs.unity3d.com/ScriptReference/ComputeBuffer.html">
 * compute buffer</a> of \p float4 created in Unity you can write directly
 * into it from CUDA.
 */
class ActionSampleVertexBuffer : public Action
{
    public:
    /**
     * @brief      Constructs a new instance of \p ActionSampleVertexBuffer
     *
     * @param      bufferPtr  The buffer pointer
     * @param[in]  size       The size of the buffer
     */
    ActionSampleVertexBuffer(void *bufferPtr, int size);

    /**
     * @brief      Register the buffer in CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    inline int Start() override;
    /**
     * @brief      Map and write the buffer in CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    inline int Update() override;
    /**
     * @brief      Unregister the buffer in CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    inline int OnDestroy() override;

    private:
    /**
     * @brief     A point to the buffer that will be written.
     */
    Buffer *_vertexBuffer;

    /**
     * @brief     Vertex array readable/writable by CUDA after map.
     */
    float4 *d_vertexArray;
};
} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleVertexBuffer
        *UNITY_INTERFACE_API
        createActionSampleVertexBufferBasic(void *bufferPtr, int size);
}
