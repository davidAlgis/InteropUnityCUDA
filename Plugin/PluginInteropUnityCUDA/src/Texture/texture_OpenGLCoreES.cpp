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
    _texTarget = _textureDepth < 2 ? GL_TEXTURE_2D : GL_TEXTURE_2D_ARRAY;

    // cast the pointer on the texture of unity to gluint
    _texUnityGL = (GLuint)(size_t)(_textureHandle);
    GL_CHECK();
    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK_RETURN(
        cudaGraphicsGLRegisterImage(&_graphicsResource, _texUnityGL, _texTarget,
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
    Log::log().debugLogError(
        "copyUnityTextureToAPITexture - Not implemented yet");
    return -1;
}

int Texture_OpenGLCoreES::copyAPITextureToUnityTexture()
{
    Log::log().debugLogError(
        "copyAPITextureToUnityTexture - Not implemented yet");
    return -1;
}

int Texture_OpenGLCoreES::generateMips()
{
    // bind the texture
    glBindTexture(_texTarget, _texUnityGL);
    GL_CHECK();
    glGenerateMipmap(_texTarget);
    GL_CHECK();
    // unbind the texture
    glBindTexture(_texTarget, 0);
    GL_CHECK();
    return SUCCESS_INTEROP_CODE;
}

#endif // #if SUPPORT_OPENGL_UNIFIED
