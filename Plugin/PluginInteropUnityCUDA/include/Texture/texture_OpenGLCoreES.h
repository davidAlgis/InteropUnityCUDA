#pragma once
#include "texture.h"

#if SUPPORT_OPENGL_UNIFIED

#include "openGL_include.h"
#include <cuda_gl_interop.h>

/**
 * @class      Texture_OpenGLCoreES
 *
 * @brief      This class handles interoperability for texture from Unity to
 *             CUDA, with OpenGL Core ES graphics API.
 *
 * @see        @ref Texture
 */
class Texture_OpenGLCoreES : public Texture
{
    public:
    Texture_OpenGLCoreES(void *textureHandle, int textureWidth,
                         int textureHeight, int textureDepth);

    /**
     * @brief      Destroys the object.
     */
    ~Texture_OpenGLCoreES();
    /**
     * @brief      Register the texture pointer given given in \ref
     * Texture_OpenGLCoreES from OpenGL Core ES to CUDA
     *
     * @return     0 if it works, otherwise something else.
     *
     *
     * @warning    Make sure that the pointer \ref Buffer::_textureHandle is
     * of type \p GLuint, otherwise it can result in a exception.
     */
    int registerTextureInCUDA() override;

    /**
     * @brief      Unregister the texture from OpenGL Core ES to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int unregisterTextureInCUDA() override;

    /**
     * @brief      Generate mips for texture.
     *
     * @return     0 if it works, otherwise something else.
     */
    int generateMips() override;

    protected:
    /**
     * @brief      Implementation of \ref Texture::copyUnityTextureToAPITexture
     * for OpenGL Core ES.
     *
     * @return     0 if it works, otherwise something else.
     */
    int copyUnityTextureToAPITexture() override;

    /**
     * @brief      Implementation of \ref Texture::copyAPITextureToUnityTexture
     * for OpenGL Core ES.
     *
     * @return     0 if it works, otherwise something else.
     */
    int copyAPITextureToUnityTexture() override;

    private:
    /**
     * @brief     A pointer toward the texture with OpenGL type.
     */
    GLuint _texUnityGL;

    /**
     * @brief     The type of the texture.
     */
    GLenum _texTarget;
};

#endif