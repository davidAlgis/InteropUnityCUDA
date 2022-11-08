#pragma once
#include "buffer.h"


void kernelCallerWriteBuffer(const dim3 dimGrid, const dim3 dimBlock);

Buffer::Buffer(void* bufferHandle, int size, int stride)
{
    _bufferHandle = bufferHandle;
    _size = size;
    _stride = stride;
    _dimBlock = { 8, 8, 1 };
    _dimGrid = { (size + _dimBlock.x - 1) / _dimBlock.x,
        1, 1};
    
}