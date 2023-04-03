#pragma once
#include "texture.h"

Texture::Texture(void *textureHandle, int textureWidth, int textureHeight,
                 int textureDepth)
{
    _textureHandle = textureHandle;
    _textureWidth = textureWidth;
    _textureHeight = textureHeight;

    if (textureDepth < 0)
    {
        Log::log().debugLogError(
            "Texture depth :" + std::to_string(textureDepth) +
            " cannot be negative !");
        return;
    }

    _textureDepth = textureDepth;

    // set a default size of grid and block to avoid calculating it each time
    // TODO : update this for texture depth
    _dimBlock = {8, 8, 1};
    _dimGrid = calculateDimGrid(_dimBlock,
                                {(unsigned int)textureWidth,
                                 (unsigned int)textureHeight,
                                 (unsigned int)textureDepth},
                                false);
    // initialize surface object
    _surfObjArray = new cudaSurfaceObject_t[textureDepth];

    for (int i = 0; i < textureDepth; i++)
    {
        _surfObjArray[i] = 0;
    }

    _graphicsResource = nullptr;
}

Texture::~Texture()
{
    delete (_surfObjArray);
}

void Texture::mapTextureToSurfaceObject()
{
    // map the resource to cuda
    CUDA_CHECK(cudaGraphicsMapResources(1, &_graphicsResource));
    for (int i = 0; i < _textureDepth; i++)
    {
        // cuda array on which the resources will be sended
        cudaArray *arrayPtr;
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
            &arrayPtr, _graphicsResource, i, 0));

        // Wrap the cudaArray in a surface object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arrayPtr;
        _surfObjArray[i] = 0;
        CUDA_CHECK(cudaCreateSurfaceObject(&_surfObjArray[i], &resDesc));
        CUDA_CHECK(cudaGetLastError());
    }
}

void Texture::unmapTextureToSurfaceObject()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_graphicsResource));
    // we destroy each surface object
    for (int i = 0; i < _textureDepth; i++)
    {
        CUDA_CHECK(cudaDestroySurfaceObject(_surfObjArray[i]));
    }
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

void *Texture::getNativeTexturePtr() const
{
    return _textureHandle;
}

cudaSurfaceObject_t *Texture::getSurfaceObjectArray() const
{
    return _surfObjArray;
}

cudaSurfaceObject_t Texture::getSurfaceObject(int indexInArray) const
{
    if (indexInArray < 0 || indexInArray > _textureDepth)
    {
        Log::log().debugLog("Could not get surface object for index " +
                            std::to_string(indexInArray) +
                            ", because it's out of bound.");
        return 0;
    }

    return _surfObjArray[indexInArray];
}
