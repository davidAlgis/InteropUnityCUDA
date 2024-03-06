#include "buffer_OpenGLCoreES.h"

#if SUPPORT_OPENGL_UNIFIED

Buffer_OpenGLCoreES::Buffer_OpenGLCoreES(void *bufferHandle, int size)
    : Buffer(bufferHandle, size)
{
}

Buffer_OpenGLCoreES::~Buffer_OpenGLCoreES()
{
    // final check to be sure there was no mistake
    GL_CHECK();
    CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Register the buffer from OpenGL to CUDA
/// </summary>
int Buffer_OpenGLCoreES::registerBufferInCUDA()
{
    // cast the pointer on the buffer of unity to gluint
    // (first to size_t to avoid a warning C4311)
    auto glBuffer = static_cast<GLuint>((size_t)(_bufferHandle));
    // register the buffer to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &_graphicsResource, glBuffer, cudaGraphicsRegisterFlagsNone));
    return SUCCESS_INTEROP_CODE;
}

int Buffer_OpenGLCoreES::unregisterBufferInCUDA()
{
    CUDA_CHECK_RETURN(cudaGraphicsUnregisterResource(_graphicsResource));
    return SUCCESS_INTEROP_CODE;
}

#endif // #if SUPPORT_OPENGL_UNIFIED
