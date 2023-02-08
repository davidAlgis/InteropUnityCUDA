#pragma once
#include "action.h"

class Texture;

namespace SampleBasic {
	class ActionSampleTextureArray : public Action {
	public:
		ActionSampleTextureArray(void* texturePtr, int width, int height, int depth);

		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		Texture* _texture;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTextureArray* UNITY_INTERFACE_API createActionSampleTextureArrayBasic(void* texturePtr, int width, int height, int depth);
}
