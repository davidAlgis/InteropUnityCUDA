#pragma once
#include "buffer_D3D11.h"
#include "buffer_OpenGLCoreES.h"
#include "renderAPI.h"
#include "texture_D3D11.h"
#include "texture_OpenGLCoreES.h"

/**
 * Simple factory template to create some vertex buffer or texture in function
 * of API type
 */
namespace Factory
{
/**
 * @brief      Will automatically create the buffer with the correct graphics
 * API
 *
 * @param      bufferHandle  The buffer handle (the native pointer given by
 * Unity)
 * @param[in]  size          The size of the buffer
 * @param[in]  apiType       The api type
 * @param      renderAPI     The \ref RenderAPI that will be used.
 *
 * @return     The buffer to use in the rest of the code.
 */
Buffer *createBuffer(void *bufferHandle, int size, UnityGfxRenderer apiType,
                     RenderAPI *renderAPI);
/**
 * @brief      Will automatically create the texture with the correct graphics
 * API
 *
 * @param      textureHandle  The texture handle (the native pointer given by
 * Unity)
 * @param[in]  textureWidth   The texture width
 * @param[in]  textureHeight  The texture height
 * @param[in]  textureDepth   The texture depth
 * @param[in]  apiType        The api type
 * @param      renderAPI     The \ref RenderAPI that will be used.
 *
 * @return     The texture to use in the rest of the code.
 */
Texture *createTexture(void *textureHandle, int textureWidth, int textureHeight,
                       int textureDepth, UnityGfxRenderer apiType,
                       RenderAPI *renderAPI);
} // namespace Factory
