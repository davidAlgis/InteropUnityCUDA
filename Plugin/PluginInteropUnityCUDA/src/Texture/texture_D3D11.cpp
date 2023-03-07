#pragma once
#include "texture_D3D11.h"

#if SUPPORT_D3D11

Texture_D3D11::Texture_D3D11(void *textureHandle, int textureWidth,
                             int textureHeight, int textureDepth)
    : Texture(textureHandle, textureWidth, textureHeight, textureDepth)
{
}

Texture_D3D11::~Texture_D3D11()
{
    CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Has to be call after the first issue plugin event
/// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
/// register a graphics resources defined from the texture openGL
/// </summary>
void Texture_D3D11::registerTextureInCUDA()
{

    Log::log().debugLog("try register");
    ID3D11Texture2D *d3dtex = (ID3D11Texture2D *)_textureHandle;
    assert(d3dtex);
    // ID3D11DeviceContext* ctx = NULL;
	// m_Device->GetImmediateContext(&ctx);
    // ID3D11Resource* tex = (ID3D11Resource*)(_textureHandle);
    // HRESULT result = tex->GetDevice()->GetDeviceRemovedReason();
    // register the texture to cuda : it initialize the _pGraphicsResource
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
        &_pGraphicsResource, d3dtex, cudaGraphicsRegisterFlagsWriteDiscard));
    Log::log().debugLog("register");
}

void Texture_D3D11::unRegisterTextureInCUDA()
{
    CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
}

#endif // #if SUPPORT_D3D11
