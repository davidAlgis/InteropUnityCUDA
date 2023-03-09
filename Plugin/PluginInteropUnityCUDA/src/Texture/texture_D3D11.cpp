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
    _texBufferInterop->Release();
    CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Has to be call after the first issue plugin event
/// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
/// register a graphics resources defined from the texture dx11
/// </summary>
void Texture_D3D11::registerTextureInCUDA()
{

    // This method initialize the buffer textures that will be registered in
    // CUDA This method will use m_device attributes in RenderAPI_D3D11, make
    // sure it has been well initialized.
    int retCodeCreate = _renderAPI->createTexture2D(
        _textureWidth, _textureHeight, _textureDepth, &_texBufferInterop);
    // we initialize
    if (retCodeCreate < 0)
    {
        Log::log().debugLogError("Could not initialize texture on DX11 for "
                                 "copy. Interoperability has failed.");
    }


    // we cast it here, to make it only once.
    _texUnityDX11 = (ID3D11Texture2D *)_textureHandle;
    // assert(_texBufferInterop);
    // D3D11_TEXTURE2D_DESC texDesc;
    // _texBufferInterop->GetDesc(&texDesc);

    // DXGI_FORMAT format = texDesc.Format;
    // Log::log().debugLog(std::to_string(format));

    // CUDA_CHECK(cudaGetLastError());
    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
        &_pGraphicsResource, _texBufferInterop, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(cudaGetLastError());
}

void Texture_D3D11::unRegisterTextureInCUDA()
{

    CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
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
