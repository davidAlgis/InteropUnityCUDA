#pragma once
#include "ActionSampleVertexBuffer.h"
#include "unityPlugin.h"
#include "VertexBuffer.h"
#include <memory>


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
		_vertexBuffer->writeBuffer(ptr, GetTime());
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
