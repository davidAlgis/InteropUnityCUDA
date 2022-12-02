#pragma once
#include "vertexBuffer.h"


VertexBuffer::VertexBuffer(void* bufferHandle, int size)
{
    _bufferHandle = bufferHandle;
    _size = size;
    _dimBlock = { 8, 8, 1 };
    _dimGrid = { (size + _dimBlock.x - 1) / _dimBlock.x,
        1, 1};
    _pGraphicsResource = nullptr;
    
}

float4* VertexBuffer::mapResources()
{
    float4* vertexPtr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource, 0));
    size_t num_bytes;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertexPtr, &num_bytes,
        _pGraphicsResource));
    return vertexPtr;
}

void VertexBuffer::unmapResources()
{
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