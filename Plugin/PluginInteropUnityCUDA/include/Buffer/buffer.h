#pragma once
#include "log.h"
#include "cudaInclude.h"


class Buffer
{
	public:
		Buffer(void* bufferHandle, int size, int stride);
		virtual void registerBufferInCUDA() = 0;

	protected:
		void* _bufferHandle;
		int _size;
		int _stride;
		cudaGraphicsResource* _pGraphicsResource;


	private:
		dim3 _dimBlock;
		dim3 _dimGrid;
};



