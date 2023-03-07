#pragma once
#include "vertex_buffer_D3D11.h"



#if SUPPORT_D3D11

VertexBuffer_D3D11::VertexBuffer_D3D11(void* bufferHandle, int size)
	: VertexBuffer(bufferHandle, size)
{}

VertexBuffer_D3D11::~VertexBuffer_D3D11()
{
	// final check to be sure there was no mistake
	CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Register the buffer from OpenGL to CUDA
/// </summary>
void VertexBuffer_D3D11::registerBufferInCUDA()
{
	Log::log().debugLog("try register");
	// register the texture to cuda : it initialize the _pGraphicsResource
	CUDA_CHECK(cudaGraphicsD3D11RegisterResource(&_pGraphicsResource, (ID3D11Resource*)_bufferHandle, cudaGraphicsRegisterFlagsWriteDiscard));
	Log::log().debugLog("register");}

void VertexBuffer_D3D11::unRegisterBufferInCUDA()
{
	CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
}


#endif // #if SUPPORT_D3D11
