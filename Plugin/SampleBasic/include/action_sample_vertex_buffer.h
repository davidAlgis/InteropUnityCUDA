#pragma once
#include "sample_kernels.cuh"
#include "action.h"
#include "vertex_buffer.h"


namespace SampleBasic {
	class ActionSampleVertexBuffer: public Action {
	public:
		ActionSampleVertexBuffer(void* bufferPtr, int size);

		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		VertexBuffer* _vertexBuffer;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleVertexBuffer* UNITY_INTERFACE_API 
		createActionSampleVertexBufferBasic(void* bufferPtr, int size);
}
