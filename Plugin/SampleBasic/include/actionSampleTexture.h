#pragma once
#include "action.h"

class Texture;

namespace SampleBasic {
	class ActionSampleTexture : public Action {
	public:
		ActionSampleTexture(void* texturePtr, int width, int height);

		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		Texture* _texture;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTexture* UNITY_INTERFACE_API createActionSampleTextureBasic(void* texturePtr, int width, int height);
}
