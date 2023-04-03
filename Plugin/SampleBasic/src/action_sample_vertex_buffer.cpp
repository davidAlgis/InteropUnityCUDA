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
		_vertexBuffer->registerBufferInCUDA();
		return 0;
	}

	int ActionSampleVertexBuffer::Update()
	{		
		float4* ptr = _vertexBuffer->mapResources<float4>();

		kernelCallerWriteBuffer(_vertexBuffer->getDimGrid(), _vertexBuffer->getDimBlock(), ptr, _vertexBuffer->getSize(), GetTime());
		cudaDeviceSynchronize();

		float4* v = (float4*)malloc(_vertexBuffer->getSize());
		cudaMemcpy(v, ptr, _vertexBuffer->getSize(), cudaMemcpyDeviceToHost);


		_vertexBuffer->unmapResources();
		delete(v);

		return 0;
	}

	inline int ActionSampleVertexBuffer::OnDestroy()
	{
		_vertexBuffer->unregisterBufferInCUDA();
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
