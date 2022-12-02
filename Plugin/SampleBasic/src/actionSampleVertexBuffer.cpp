#pragma once
#include "ActionSampleVertexBuffer.h"
#include "unityPlugin.h"
#include "VertexBuffer.h"

void kernelCallerWriteBuffer(const dim3 dimGrid, const dim3 dimBlock, float4* vertexPtr, const int size, const float time);

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
		float4* ptr = _vertexBuffer->mapResources();

		kernelCallerWriteBuffer(_vertexBuffer->getDimGrid(), _vertexBuffer->getDimBlock(), ptr, _vertexBuffer->getSize(), GetTime());
		cudaDeviceSynchronize();

		float4* v = (float4*)malloc(_vertexBuffer->getSize());
		cudaMemcpy(v, ptr, _vertexBuffer->getSize(), cudaMemcpyDeviceToHost);
		Log::log().debugLog(std::to_string(v[2].x));


		_vertexBuffer->unmapResources();
		return 0;
	}

	inline int ActionSampleVertexBuffer::OnDestroy()
	{
		_vertexBuffer->unRegisterBufferInCUDA();
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
