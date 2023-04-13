#pragma once
// Direct3D 11 implementation of Vertex Buffer API
#include "vertex_buffer.h"


#if SUPPORT_D3D11

#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include "IUnityGraphicsD3D11.h"

class VertexBuffer_D3D11 : public VertexBuffer
{
public:
	VertexBuffer_D3D11(void* bufferHandle, int size);
	~VertexBuffer_D3D11();
	virtual int registerBufferInCUDA();
	virtual int unregisterBufferInCUDA();

};

#endif
