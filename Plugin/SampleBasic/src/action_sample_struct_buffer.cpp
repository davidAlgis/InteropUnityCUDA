#pragma once
#include "action_sample_struct_buffer.h"
#include "unity_plugin.h"

namespace SampleBasic
{

ActionSampleStructBuffer::ActionSampleStructBuffer(void *bufferPtr, int size)
{
    _structBuffer = CreateBufferInterop(bufferPtr, size);
}

inline int ActionSampleStructBuffer::Start()
{
    int ret = _structBuffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

int ActionSampleStructBuffer::Update()
{
    SampleStructInterop *ptr = nullptr;
    int ret = _structBuffer->mapResources<SampleStructInterop>(&ptr);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");

    kernelCallerWriteBufferStruct(_structBuffer->getDimGrid(),
                                  _structBuffer->getDimBlock(), ptr,
                                  _structBuffer->getSize(), GetTimeInterop());
    cudaDeviceSynchronize();

    ret = _structBuffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");

    return 0;
}

inline int ActionSampleStructBuffer::OnDestroy()
{
    int ret = _structBuffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the struct buffer in CUDA. Abort ActionSampleStructBuffer !");
    return 0;
}

} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleStructBuffer
        *UNITY_INTERFACE_API
        createActionSampleStructBufferBasic(void *bufferPtr, int size)
    {
        return (new SampleBasic::ActionSampleStructBuffer(bufferPtr, size));
    }
} // extern "C"
