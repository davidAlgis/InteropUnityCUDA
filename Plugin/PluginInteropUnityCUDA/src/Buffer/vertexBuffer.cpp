#pragma once
#include "vertexBuffer.h"



void kernelCallerWriteBuffer(const dim3 dimGrid, const dim3 dimBlock, float4* vertexPtr, const int size, const float time);

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

void VertexBuffer::writeBuffer(float4* vertexPtr, const float time)
{
    kernelCallerWriteBuffer(_dimGrid, _dimBlock, vertexPtr, _size, time);
    cudaDeviceSynchronize();
    float4* v = (float4*)malloc(_size);
    cudaMemcpy(v, vertexPtr, _size, cudaMemcpyDeviceToHost);
    Log::log().debugLog(std::to_string(v[2].x));

}

void VertexBuffer::unmapResources()
{
    cudaGraphicsUnmapResources(1, &_pGraphicsResource, 0);
}
