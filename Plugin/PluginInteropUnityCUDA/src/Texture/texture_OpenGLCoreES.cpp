#pragma once
#include "texture_OpenGLCoreES.h"

#if SUPPORT_OPENGL_UNIFIED

Texture_OpenGLCoreES::Texture_OpenGLCoreES(void *textureHandle,
                                           int textureWidth, int textureHeight,
                                           int textureDepth)
    : Texture(textureHandle, textureWidth, textureHeight, textureDepth)
{
}

Texture_OpenGLCoreES::~Texture_OpenGLCoreES()
{
    GL_CHECK();
    CUDA_CHECK(cudaGetLastError());
};


int Texture_OpenGLCoreES::registerTextureInCUDA()
{
    // if depth is < 2 it's a texture2D, else it's a texture2DArray
    GLenum target = _textureDepth < 2 ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY;

    // cast the pointer on the texture of unity to gluint
    GLuint gltex = (GLuint)(size_t)(_textureHandle);
    // glBindTexture(target, gltex);
    GL_CHECK();
    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK_RETURN(
        cudaGraphicsGLRegisterImage(&_graphicsResource, gltex, target,
                                    cudaGraphicsRegisterFlagsWriteDiscard));
    return SUCCESS_INTEROP_CODE;
}

int Texture_OpenGLCoreES::unregisterTextureInCUDA()
{
    CUDA_CHECK_RETURN(cudaGraphicsUnregisterResource(_graphicsResource));
    return SUCCESS_INTEROP_CODE;
}

int Texture_OpenGLCoreES::copyUnityTextureToAPITexture()
{
    return SUCCESS_INTEROP_CODE;
}

int Texture_OpenGLCoreES::copyAPITextureToUnityTexture()
{
    return SUCCESS_INTEROP_CODE;
}



#endif // #if SUPPORT_OPENGL_UNIFIED
