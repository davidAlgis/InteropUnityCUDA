#pragma once
#include "log.h"
#include "cudaInclude.h"


class UNITY_INTERFACE_EXPORT VertexBuffer
{
	public:
		VertexBuffer(void* bufferHandle, int size);
		virtual void registerBufferInCUDA() = 0;
		virtual void unRegisterBufferInCUDA() = 0;
		float4* mapResources();
		void writeBuffer(float4* vertexPtr, const float time);
		void unmapResources();

	protected:
		void* _bufferHandle;
		int _size;
		cudaGraphicsResource* _pGraphicsResource;


	private:
		dim3 _dimBlock;
		dim3 _dimGrid;
};



