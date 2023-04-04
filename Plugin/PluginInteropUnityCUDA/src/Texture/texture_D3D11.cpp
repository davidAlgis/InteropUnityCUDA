#pragma once
#include "texture_D3D11.h"

#if SUPPORT_D3D11

Texture_D3D11::Texture_D3D11(void *textureHandle, int textureWidth,
                             int textureHeight, int textureDepth,
                             RenderAPI *renderAPI)
    : Texture(textureHandle, textureWidth, textureHeight, textureDepth)
{

    // we need the render api associated to dx11, because we need device to
    // initialize texture
    _renderAPI = (RenderAPI_D3D11 *)renderAPI;
}

Texture_D3D11::~Texture_D3D11()
{
    // _texBufferInterop->Release();
    CUDA_CHECK(cudaGetLastError());
};

void Texture_D3D11::registerTextureInCUDA()
{
    // texture2D and texture2D array are ID3D11Texture2D in Unity for DX11
    ID3D11Texture2D *texUnityDX11 = (ID3D11Texture2D *)_textureHandle;

    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
        &_graphicsResource, texUnityDX11, cudaGraphicsRegisterFlagsNone));

    CUDA_CHECK(cudaGetLastError());
}

void Texture_D3D11::unregisterTextureInCUDA()
{

    CUDA_CHECK(cudaGraphicsUnregisterResource(_graphicsResource));
}

void Texture_D3D11::copyUnityTextureToAPITexture()
{
    _renderAPI->copyTextures2D(_texBufferInterop, _texUnityDX11);
}

void Texture_D3D11::copyAPITextureToUnityTexture()
{
    _renderAPI->copyTextures2D(_texUnityDX11, _texBufferInterop);
}

#endif // #if SUPPORT_D3D11
