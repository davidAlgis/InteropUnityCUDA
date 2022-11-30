#pragma once
#include "action.h"
#include "texture.h"

namespace SampleBasic {
	class ActionSampleTexture: public Action {
	public:
		ActionSampleTexture(void* texturePtr, int width, int height);
		~ActionSampleTexture();
		ActionSampleTexture(const ActionSampleTexture&) = default;
		ActionSampleTexture(ActionSampleTexture&&) = default;
		ActionSampleTexture& operator=(const ActionSampleTexture&) = default;
		ActionSampleTexture& operator=(ActionSampleTexture&&) = default;

		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		Texture* _texture;
		bool _hasBeenRegistered;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTexture* UNITY_INTERFACE_API createActionSampleTextureBasic(void* texturePtr, int width, int height);
}
