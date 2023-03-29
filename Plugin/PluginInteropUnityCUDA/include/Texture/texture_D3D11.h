#pragma once
#include "texture.h"

#if SUPPORT_D3D11

#include "d3d11.h"
#include "renderAPI_D3D11.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>
#include "IUnityGraphicsD3D11.h"

/// <summary>
/// This class handles interoperability for texture from Unity to CUDA, with
/// dx11 graphics API. With dx11 interoperability work differently, texture
/// created in Unity, cannot be directly registered in CUDA, therefore we need
/// to create another texture
/// (_texBufferInterop) that will be used as a buffer. We will copy the content
/// of the texture created in Unity and then we will registered this new texture
/// in CUDA see issue #2 on github for more details
/// </summary>
template <class T> class Texture_D3D11 : public Texture<T>
{
    public:
    Texture_D3D11(void *textureHandle, int textureWidth, int textureHeight,
                  int textureDepth, RenderAPI *renderAPI)
        : Texture(textureHandle, textureWidth, textureHeight, textureDepth)
    {

        // we need the render api associated to dx11, because we need device to
        // initialize texture
        _renderAPI = (RenderAPI_D3D11 *)renderAPI;
    }

    ~Texture_D3D11()
    {
        // _texBufferInterop->Release();
        CUDA_CHECK(cudaGetLastError());
    };

    /// <summary>
    /// Has to be call after the first issue plugin event
    /// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
    /// register a graphics resources defined from the texture dx11
    /// </summary>
    virtual void registerTextureInCUDA()
    {
        // texture2D and texture2D array are ID3D11Texture2D in Unity for DX11
        ID3D11Texture2D *texUnityDX11 = (ID3D11Texture2D *)_textureHandle;

        // register the texture to cuda : it initialize the _pGraphicsResource
        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
            &_pGraphicsResource, texUnityDX11, cudaGraphicsRegisterFlagsNone));

        CUDA_CHECK(cudaGetLastError());
        createSurfaceWrapper_DX11(_surfaceWrapper, _pGraphicsResource);
    }

    virtual void unRegisterTextureInCUDA()
    {
        deleteSurfaceWrapper(_surfaceWrapper);
        CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
    }

    virtual void copyUnityTextureToAPITexture()
    {
        _renderAPI->copyTextures2D(_texBufferInterop, _texUnityDX11);
    }

    virtual void copyAPITextureToUnityTexture()
    {
        _renderAPI->copyTextures2D(_texUnityDX11, _texBufferInterop);
    }

    private:
    ID3D11Texture2D *_texBufferInterop{};
    ID3D11Texture2D *_texUnityDX11{};
    RenderAPI_D3D11 *_renderAPI;
};

#endif