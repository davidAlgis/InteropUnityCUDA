#pragma once
#include "vertexBuffer.h"


VertexBuffer::VertexBuffer(void* bufferHandle, int size)
{
    _bufferHandle = bufferHandle;
    _size = size;
    // set a default size of grid and block to avoid calculating it each time
    _dimBlock = { 8, 1, 1 };
    _dimGrid = { (size + _dimBlock.x - 1) / _dimBlock.x,
        1, 1};
    _pGraphicsResource = nullptr;
    
}

float4* VertexBuffer::mapResources()
{
    // map resource
    CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource, 0));
    // pointer toward an array of float4 on device memory : the compute buffer
    float4* vertexPtr;
    // number of bytes that has been readed
    size_t numBytes;
    // map the resources on a float4 array that can be modify on device
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertexPtr, &numBytes,
        _pGraphicsResource));
    return vertexPtr;
}

void VertexBuffer::unmapResources()
{
    // unmap the resources
    cudaGraphicsUnmapResources(1, &_pGraphicsResource, 0);
}

int VertexBuffer::getSize() const
{
    return _size;
}

dim3 VertexBuffer::getDimBlock() const
{
    return _dimBlock;
}

dim3 VertexBuffer::getDimGrid() const
{
    return _dimGrid;
}