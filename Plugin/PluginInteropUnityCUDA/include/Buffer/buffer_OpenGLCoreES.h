#pragma once
#include "buffer.h"

#if SUPPORT_OPENGL_UNIFIED

#include "openGL_include.h"
#include <cuda_gl_interop.h>

/**
 * @class      Buffer_OpenGLCoreES
 *
 * @brief      This class is a implementation of \ref Buffer for OpenGL ES
 * graphics API. Indeed, the register functions depends on the graphics API.
 * Supports several flavors: Core, ES2, ES3
 *
 * @see \ref Buffer
 */
class Buffer_OpenGLCoreES : public Buffer
{
    public:
    /**
     * @brief      Constructs a new instance of Buffer for OpenGL Core ES
     * graphics API
     *
     * @param      bufferHandle  A native pointer of a buffer created in Unity
     * with OpenGL Core ES graphics API. It must be of type \p GLuint.
     *
     * @param[in]  size          The size of the buffer
     *
     * @warning    If the pointer is not of type \p GLuint, it can result
     * in a exception.
     */
    Buffer_OpenGLCoreES(void *bufferHandle, int size);
    /**
     * @brief      Destroys the object.
     */
    ~Buffer_OpenGLCoreES();

    /**
     * @brief      Register the buffer given in \ref Buffer_D3D11 from OpenGL
     * Core ES to CUDA
     *
     * @return     0 if it works, otherwise something else.
     *
     *
     * @warning    Make sure that the pointer \ref Buffer::_bufferHandle is
     * of type \p GLuint, otherwise it can result in a exception.
     */
    int registerBufferInCUDA() override;

    /**
     * @brief      Unregister the buffer from OpenGL Core ES to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int unregisterBufferInCUDA() override;
};

#endif
