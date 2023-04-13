#pragma once
#include "action_sample_vertex_buffer.h"
#include "unity_plugin.h"


namespace SampleBasic {

	ActionSampleVertexBuffer::ActionSampleVertexBuffer(void* bufferPtr, int size) : Action()
	{
		_vertexBuffer = CreateVertexBufferInterop(bufferPtr, size);
	}

	inline int ActionSampleVertexBuffer::Start()
	{
		int ret = _vertexBuffer->registerBufferInCUDA();
		GRUMBLE(ret, "There has been an error during the registration of "
            "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");
		return 0;
	}

	int ActionSampleVertexBuffer::Update()
	{		
		float4* ptr = nullptr;
		int ret = _vertexBuffer->mapResources<float4>(&ptr);
		GRUMBLE(ret, "There has been an error during the map of "
            "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");

		kernelCallerWriteBuffer(_vertexBuffer->getDimGrid(), _vertexBuffer->getDimBlock(), ptr, _vertexBuffer->getSize(), GetTime());
		cudaDeviceSynchronize();

		float4* v = (float4*)malloc(_vertexBuffer->getSize());
		CUDA_CHECK_RETURN(cudaMemcpy(v, ptr, _vertexBuffer->getSize(), cudaMemcpyDeviceToHost));


		ret = _vertexBuffer->unmapResources();
		GRUMBLE(ret, "There has been an error during the unmap of "
            "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");

		delete(v);

		return 0;
	}

	inline int ActionSampleVertexBuffer::OnDestroy()
	{
		int ret = _vertexBuffer->unregisterBufferInCUDA();
		GRUMBLE(ret, "There has been an error during the unregistration of "
            "the vertex buffer in CUDA. Abort ActionSampleVertexBuffer !");
		return 0;
	}

} // namespace SampleBasic


extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleVertexBuffer* UNITY_INTERFACE_API 
		createActionSampleVertexBufferBasic(void* bufferPtr, int size)
	{
		return (new SampleBasic::ActionSampleVertexBuffer(bufferPtr, size));
	}
} // extern "C"
