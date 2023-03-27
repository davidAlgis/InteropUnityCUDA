#pragma once
#include "sample_kernels.cuh"
#include "action.h"
#include "texture.h"
#include "cuda_include.h"


namespace SampleBasic {
	class ActionSampleTextureArray : public Action {
	public:
		ActionSampleTextureArray(void* texturePtr, int width, int height, int depth);
		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		Texture<float4>* _texture;
		cudaSurfaceObject_t _surf;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTextureArray* UNITY_INTERFACE_API createActionSampleTextureArrayBasic(void* texturePtr, int width, int height, int depth);
}
