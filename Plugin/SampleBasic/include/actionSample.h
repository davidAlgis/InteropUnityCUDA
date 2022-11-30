#pragma once
#include "action.h"
#include "texture.h"

namespace SampleBasic {
	class ActionSample : public Action {
	public:
		ActionSample(void* texturePtr, int width, int height);
		~ActionSample();
		ActionSample(const ActionSample&) = default;
		ActionSample(ActionSample&&) = default;
		ActionSample& operator=(const ActionSample&) = default;
		ActionSample& operator=(ActionSample&&) = default;

		inline int Start() override;
		inline int Update() override;
		inline int OnDestroy() override;

	private:
		Texture* _texture;
		bool _hasBeenRegistered;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSample* UNITY_INTERFACE_API createActionSampleBasic(void* texturePtr, int width, int height);
}
