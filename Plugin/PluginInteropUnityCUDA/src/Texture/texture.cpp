#pragma once
#include "texture.h"



Texture::Texture(void* textureHandle, int textureWidth, int textureHeight, int textureDepth)
{
    _textureHandle = textureHandle;
    _textureWidth = textureWidth;
    _textureHeight = textureHeight;
    _textureDepth = textureDepth;
    // set a default size of grid and block to avoid calculating it each time
    // TODO : update this for texture depth
    _dimBlock = { 8, 8, 1 };
    _dimGrid = calculateDimGrid(_dimBlock,
                                {(unsigned int)textureWidth,
                                 (unsigned int)textureHeight,
                                 (unsigned int)_textureDepth},
        false);
    _pGraphicsResource = nullptr;
}

/// <summary>
/// Use the graphics resources registered by graphics API
/// to map it to cuda array and then create a surface object
/// that will be used in kernel
/// </summary>
/// <returns></returns>
cudaSurfaceObject_t Texture::mapTextureToSurfaceObject(int indexInArray)
{
    // map the resource to cuda
    CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource));
    // cuda array on which the resources will be sended 
    cudaArray* arrayPtr;
    //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&arrayPtr, _pGraphicsResource, indexInArray, 0));

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


cudaTextureObject_t Texture::mapTextureToTextureObject(int indexInArray)
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
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    CUDA_CHECK(cudaGetLastError());
    return texObj;
}



void Texture::unMapTextureToSurfaceObject(cudaSurfaceObject_t& inputSurfObj)
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
    CUDA_CHECK(cudaDestroySurfaceObject(inputSurfObj));
    CUDA_CHECK(cudaGetLastError());
}

void Texture::unMapTextureToTextureObject(cudaTextureObject_t &texObj)
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaGetLastError());
}

dim3 Texture::getDimBlock() const
{
    return _dimBlock;
}

dim3 Texture::getDimGrid() const
{
    return _dimGrid;
}

int Texture::getWidth() const
{
    return _textureWidth;
}

int Texture::getHeight() const
{
    return _textureHeight;
}

int Texture::getDepth() const
{
    return _textureDepth;
}

void* Texture::getNativeTexturePtr() const
{
    return _textureHandle;
}




