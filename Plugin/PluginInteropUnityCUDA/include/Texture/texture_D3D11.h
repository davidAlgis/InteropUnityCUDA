#pragma once
#include "texture.h"

#if SUPPORT_D3D11

#include "d3d11.h"
#include "IUnityGraphicsD3D11.h"
#include "renderAPI_D3D11.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>

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
        _surfObjectsArray = new cudaSurfaceObject_t[textureDepth];
        for(int i=0; i<textureDepth;i++)
        {
            _surfObjectsArray[i] = 0;
        }
    }

    ~Texture_D3D11()
    {
        // _texBufferInterop->Release();
        CUDA_CHECK(cudaGetLastError());
        delete(_surfObjectsArray);
    };

    /// <summary>
    /// Has to be call after the first issue plugin event
    /// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
    /// register a graphics resources defined from the texture dx11
    /// </summary>
    virtual void registerTextureInCUDA()
    {
        Log::log().debugLog("register");
        // texture2D and texture2D array are ID3D11Texture2D in Unity for DX11
        ID3D11Texture2D *texUnityDX11 = (ID3D11Texture2D *)_textureHandle;

        // register the texture to cuda : it initialize the _pGraphicsResource
        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(
            &_pGraphicsResource, texUnityDX11, cudaGraphicsRegisterFlagsNone));

        CUDA_CHECK(cudaGetLastError());
        kernelCallerCreateSurfaceWrapper_DX11(_surfaceWrapper,
                                              _pGraphicsResource, _textureDepth);
    }

    virtual void mapTextureToSurfaceObject()
    {
        for (int indexInArray = 0; indexInArray < _textureDepth; indexInArray++)
        {
             // map the resource to cuda
            CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource));
            // cuda array on which the resources will be sended
            cudaArray *arrayPtr;
            // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
            CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
                &arrayPtr, _pGraphicsResource, indexInArray, 0));

            // Wrap the cudaArray in a surface object
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = arrayPtr;
            CUDA_CHECK(cudaCreateSurfaceObject(&_surfObjectsArray[indexInArray], &resDesc));
            CUDA_CHECK(cudaGetLastError());
        }

        Surface_D3D11<T>* surfWrapperD3D11 = (*(Surface_D3D11<T>**)_surfaceWrapper);
        CUDA_CHECK(cudaMemcpy(surfWrapperD3D11->surfObjectsArray, _surfObjectsArray, _textureDepth*sizeof(cudaSurfaceObject_t),cudaMemcpyHostToDevice));
    }

    virtual void unRegisterTextureInCUDA()
    {
        kernelCallerDeleteSurfaceWrapper(_surfaceWrapper);
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
    cudaSurfaceObject_t *_surfObjectsArray;
    ID3D11Texture2D *_texBufferInterop{};
    ID3D11Texture2D *_texUnityDX11{};
    RenderAPI_D3D11 *_renderAPI;
};

#endif