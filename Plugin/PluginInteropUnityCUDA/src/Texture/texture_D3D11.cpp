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

int Texture_D3D11::registerTextureInCUDA()
{
    // texture2D and texture2D array are ID3D11Texture2D in Unity for DX11
    ID3D11Texture2D *texUnityDX11 = (ID3D11Texture2D *)_textureHandle;

    D3D11_TEXTURE2D_DESC texDesc;
    texUnityDX11->GetDesc(&texDesc);

    DXGI_FORMAT format = texDesc.Format;

    // We check if the format is correct see
    // https://github.com/davidAlgis/InteropUnityCUDA/issues/2
    if (format == DXGI_FORMAT_R8G8B8A8_TYPELESS ||
        format == DXGI_FORMAT_R32G32B32_TYPELESS ||
        format == DXGI_FORMAT_R32G32B32A32_TYPELESS ||
        format == DXGI_FORMAT_R16G16B16A16_TYPELESS ||
        format == DXGI_FORMAT_R32G8X24_TYPELESS ||
        format == DXGI_FORMAT_R32G32_TYPELESS ||
        format == DXGI_FORMAT_R8_TYPELESS)
    {
        Log::log().debugLogError(
            "Texture of type " + std::to_string(format) +
            " cannot be registered in CUDA." +
            " It may comes from the fact that you can\'t used RenderTexture for "
            "DX11 but only Texture2D.");
        return -1;
    }

    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK_RETURN(cudaGraphicsD3D11RegisterResource(
        &_graphicsResource, texUnityDX11, cudaGraphicsRegisterFlagsNone));

    return SUCCESS_INTEROP_CODE;
}

int Texture_D3D11::unregisterTextureInCUDA()
{
    CUDA_CHECK_RETURN(cudaGraphicsUnregisterResource(_graphicsResource));
    return SUCCESS_INTEROP_CODE;
}

int Texture_D3D11::copyUnityTextureToAPITexture()
{
    _renderAPI->copyTextures2D(_texBufferInterop, _texUnityDX11);
    return SUCCESS_INTEROP_CODE;
}

int Texture_D3D11::copyAPITextureToUnityTexture()
{
    _renderAPI->copyTextures2D(_texUnityDX11, _texBufferInterop);
    return SUCCESS_INTEROP_CODE;
}

#endif // #if SUPPORT_D3D11
