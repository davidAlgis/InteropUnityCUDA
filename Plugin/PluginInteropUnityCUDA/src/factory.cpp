#include "factory.h"

namespace Factory
{
VertexBuffer *createBuffer(void *bufferHandle, int size,
                           UnityGfxRenderer apiType, RenderAPI* renderAPI)
{
    VertexBuffer *buffer = NULL;

#if SUPPORT_D3D11
    if (apiType == kUnityGfxRendererD3D11)
    {
        buffer = new VertexBuffer_D3D11(bufferHandle, size);
    }
#endif // if SUPPORT_D3D11
    // if SUPPORT_OPENGL_UNIFIED
#if SUPPORT_OPENGL_UNIFIED
    if (apiType == kUnityGfxRendererOpenGLCore ||
        apiType == kUnityGfxRendererOpenGLES20 ||
        apiType == kUnityGfxRendererOpenGLES30)
    {
        buffer = new VertexBuffer_OpenGLCoreES(bufferHandle, size);
    }
#endif

    // Unknown or unsupported graphics API
    return buffer;
}

Texture *createTexture(void *textureHandle, int textureWidth, int textureHeight,
                       int textureDepth, UnityGfxRenderer apiType, RenderAPI* renderAPI)
{
    Texture *texture = NULL;
#if SUPPORT_D3D11
    if (apiType == kUnityGfxRendererD3D11)
    {
        texture = new Texture_D3D11(textureHandle, textureWidth,
                                           textureHeight, textureDepth, renderAPI);
    }
#endif 
#if SUPPORT_OPENGL_UNIFIED
    if (apiType == kUnityGfxRendererOpenGLCore ||
        apiType == kUnityGfxRendererOpenGLES20 ||
        apiType == kUnityGfxRendererOpenGLES30)
    {
        texture = new Texture_OpenGLCoreES(textureHandle, textureWidth,
                                           textureHeight, textureDepth);
    }
#endif

    // will be NULL is unknown or unsupported graphics API
    return texture;
}

} // namespace Factory