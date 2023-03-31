#pragma once
#include "surface_wrapper.h"
#include "surface_wrapper_device_allocator.cuh"
#include "IUnityGraphics.h"
#include "cuda_include.h"
#include "log.h"

template <class T> class Texture
{
    public:
    /// <summary>
    /// Constructor of texture
    /// </summary>
    /// <param name="textureHandle">A pointer of texture with float4 that has
    /// been generated with Unity (see function GetNativeTexturePtr
    /// https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html)
    /// </param> <param name="textureWidth">the width of the texture</param>
    /// <param name="textureHeight">the height of the texture</param>
    /// <param name="textureDepth">the depth of the texture</param>
    UNITY_INTERFACE_EXPORT Texture(void *textureHandle, int textureWidth,
                                   int textureHeight, int textureDepth)
    {
        _textureHandle = textureHandle;
        _textureWidth = textureWidth;
        _textureHeight = textureHeight;
        _textureDepth = textureDepth;
        // set a default size of grid and block to avoid calculating it each
        // time
        // TODO : update this for texture depth
        _dimBlock = {8, 8, 1};
        _dimGrid = calculateDimGrid(_dimBlock,
                                    {(unsigned int)textureWidth,
                                     (unsigned int)textureHeight,
                                     (unsigned int)textureDepth},
                                    false);
        _pGraphicsResource = nullptr;
        CUDA_CHECK(cudaMalloc(&_surfaceWrapper, sizeof(SurfaceWrapper<T>**)));

    }

    /// <summary>
    /// Register the texture in CUDA, this has to be override because it depends
    /// on the graphics api
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void registerTextureInCUDA() = 0;

    /// <summary>
    /// Unregister the texture in CUDA, this has to be override because it
    /// depends on the graphics api
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void unRegisterTextureInCUDA() = 0;

    /// <summary>
    /// For some API (DX11) CUDA cannot edit the texture created by Unity
    /// therefore, we have to create a new texture that will used as a buffer
    /// between the unity texture and the surface object that is modify by CUDA
    /// For these API, this function will copy the content of the unity texture
    /// to this buffer, for the other API. It'll do nothing.
    /// If Unity texture has been modify in Unity, you have to do the copy
    /// before reading it in CUDA. It's not necessary if you only write into the
    /// texture in CUDA, or if the texture has not been modify in Unity. Tips :
    /// not necessary for write only in CUDA or read only in Unity
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void copyUnityTextureToAPITexture() = 0;

    /// <summary>
    /// For some API (DX11) CUDA cannot edit the texture created by Unity
    /// therefore, we have to create a new texture that will used as a buffer
    /// between the unity texture and the surface object that is modify by CUDA
    /// For these API, this function will copy the content of the buffer texture
    /// to the unity texture, for the other API. It'll do nothing.
    /// If API texture has been modify by, you have to do the copy before
    /// reading it in Unity. It's not necessary if you only read into the
    /// texture in CUDA, or if the texture is only write only in Unity. Tips :
    /// not necessary for read only in CUDA or write only in Unity
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void copyAPITextureToUnityTexture() = 0;

    /// <summary>
    /// Map a cuda array to the graphics resources and wrap it into a surface
    /// object of cuda
    /// </summary>
    /// <param name="indexInArray"> Array index for array textures or cubemap
    /// face index as defined by cudaGraphicsCubeFace for cubemap textures for
    /// the subresource to access </param> <returns>a cuda surface object on
    /// device memory and which can be edited in cuda</returns>
    UNITY_INTERFACE_EXPORT cudaSurfaceObject_t
    mapTextureToSurfaceObject(int indexInArray = 0)
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
        cudaSurfaceObject_t inputSurfObj = 0;
        CUDA_CHECK(cudaCreateSurfaceObject(&inputSurfObj, &resDesc));
        CUDA_CHECK(cudaGetLastError());
        return inputSurfObj;
    }


    /// <summary>
    /// Unmap the cuda array from graphics resources and destroy surface object
    /// This function will wait for all previous GPU activity to complete
    /// </summary>
    /// <param name="inputSurfObj">the surface object that has been created with
    /// <c>mapTextureToSurfaceObject</c> function</param>
    UNITY_INTERFACE_EXPORT void unMapTextureToSurfaceObject(
        cudaSurfaceObject_t &inputSurfObj)
    {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
        CUDA_CHECK(cudaDestroySurfaceObject(inputSurfObj));
        CUDA_CHECK(cudaGetLastError());
    }

    UNITY_INTERFACE_EXPORT void unMapTextureToTextureObject(
        cudaTextureObject_t &texObj)
    {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
        CUDA_CHECK(cudaDestroyTextureObject(texObj));
        CUDA_CHECK(cudaGetLastError());
    }

    UNITY_INTERFACE_EXPORT dim3 getDimBlock() const
    {
        return _dimBlock;
    }

    UNITY_INTERFACE_EXPORT dim3 getDimGrid() const
    {
        return _dimGrid;
    }

    UNITY_INTERFACE_EXPORT int getWidth() const
    {
        return _textureWidth;
    }

    UNITY_INTERFACE_EXPORT int getHeight() const
    {
        return _textureHeight;
    }

    UNITY_INTERFACE_EXPORT int getDepth() const
    {
        return _textureDepth;
    }

    UNITY_INTERFACE_EXPORT void *getNativeTexturePtr() const
    {
        return _textureHandle;
    }

    UNITY_INTERFACE_EXPORT cudaTextureObject_t
    mapTextureToTextureObject(int indexInArray = 0)
    {
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

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 1;

            cudaTextureObject_t texObj = 0;
            CUDA_CHECK(
                cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
            CUDA_CHECK(cudaGetLastError());
            return texObj;
        }
    }

    protected:
    // Pointer to the texture created in Unity
    void *_textureHandle;
    // width of the texture
    int _textureWidth;
    // height of the texture
    int _textureHeight;
    // depth of the texture <2 <=> to Texture2D; >1 <=> Texture2DArray
    int _textureDepth;
    // Resource that can be used to retrieve the surface object for CUDA
    cudaGraphicsResource *_pGraphicsResource;

    SurfaceWrapper<T>** _surfaceWrapper;

    private:
    dim3 _dimBlock;
    dim3 _dimGrid;
};
