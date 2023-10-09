#pragma once
#include "action.h"
#include "buffer.h"
#include "sample_kernels.cuh"

namespace SampleBasic
{
class ActionSampleVertexBuffer : public Action
{
    public:
    ActionSampleVertexBuffer(void *bufferPtr, int size);

    inline int Start() override;
    inline int Update() override;
    inline int OnDestroy() override;

    private:
    Buffer *_vertexBuffer;
};
} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleVertexBuffer
        *UNITY_INTERFACE_API
        createActionSampleVertexBufferBasic(void *bufferPtr, int size);
}
