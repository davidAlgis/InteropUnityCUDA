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
    _dimGrid = calculateDimGrid(
        _dimBlock, {(unsigned int)textureWidth, (unsigned int)textureHeight, 1},
        false);
    dim3 dimGrid = { (textureWidth + _dimBlock.x - 1) / _dimBlock.x,
        (textureHeight + _dimBlock.y - 1) / _dimBlock.y, 1};
    Log::log().debugLog("(" + std::to_string(dimGrid.x) + ", " +
                        std::to_string(dimGrid.y) + ", " +
                        std::to_string(dimGrid.z) + ")");
    Log::log().debugLog("(" + std::to_string(_dimGrid.x) + ", " +
                        std::to_string(_dimGrid.y) + ", " +
                        std::to_string(_dimGrid.z) + ")");
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
    cudaGraphicsMapResources(1, &_pGraphicsResource);
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
    return inputSurfObj;
}


void Texture::unMapTextureToSurfaceObject(cudaSurfaceObject_t& inputSurfObj)
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_pGraphicsResource));
    CUDA_CHECK(cudaDestroySurfaceObject(inputSurfObj));
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




