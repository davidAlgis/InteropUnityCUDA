#include "action_sample_vertex_buffer.h"
#include "unity_plugin.h"

namespace SampleBasic
{

ActionSampleVertexBuffer::ActionSampleVertexBuffer(void *bufferPtr, int size)
{
    _vertexBuffer = CreateBufferInterop(bufferPtr, size);
}

inline int ActionSampleVertexBuffer::Start()
{
    int ret = _vertexBuffer->registerBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");

    // We map the ressources to d_vertexArray, which is a "classical" device
    // array readable/writable by CUDA. We can apply the mapResources in Start
    // and unmapResources in Destroy, because the vertexBuffer is only read in
    // Unity. Indeed, if Unity writes into the vertexBuffer at the same time
    // that CUDA do, without mapping it before use each frame, it makes Unity
    // crash.
    // Therefore, if you want to write in the vertex buffer with Unity and CUDA
    // at update, don't forget to map and unmap the array at update too !
    // However, this function caused an large overhead, especially
    // for OpenGL graphics API, that's why we advise to make all the write in
    // CUDA parts.
    ret = _vertexBuffer->mapResources<float4>(&d_vertexArray);
    GRUMBLE(ret, "There has been an error during the map of "
                 "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");

    return 0;
}

int ActionSampleVertexBuffer::Update()
{
    kernelCallerWriteBuffer(_vertexBuffer->getDimGrid(),
                            _vertexBuffer->getDimBlock(), d_vertexArray,
                            _vertexBuffer->getSize(), GetTimeInterop());

    return 0;
}

inline int ActionSampleVertexBuffer::OnDestroy()
{
    int ret = _vertexBuffer->unmapResources();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");
    ret = _vertexBuffer->unregisterBufferInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");
    return 0;
}

} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleVertexBuffer
        *UNITY_INTERFACE_API
        createActionSampleVertexBufferBasic(void *bufferPtr, int size)
    {
        return (new SampleBasic::ActionSampleVertexBuffer(bufferPtr, size));
    }
} // extern "C"
