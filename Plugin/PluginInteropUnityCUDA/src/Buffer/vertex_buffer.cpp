#pragma once
#include "vertex_buffer.h"


VertexBuffer::VertexBuffer(void* bufferHandle, int size)
{
    _bufferHandle = bufferHandle;
    _size = size;
    // set a default size of grid and block to avoid calculating it each time
    _dimBlock = { 8, 1, 1 };
    _dimGrid = { (size + _dimBlock.x - 1) / _dimBlock.x,
        1, 1};
    _graphicsResource = nullptr;
    
}



int VertexBuffer::unmapResources()
{
    // unmap the resources
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &_graphicsResource, 0));
    return SUCCESS_INTEROP_CODE;
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