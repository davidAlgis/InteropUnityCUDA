#pragma once
#include "action.h"
#include "texture.h"

namespace SampleBasic {
	class ActionSample : public Action {
	public:
		ActionSample() : Action() {}
		ActionSample(const ActionSample&) = default;
		ActionSample(ActionSample&&) = default;
		ActionSample& operator=(const ActionSample&) = default;
		ActionSample& operator=(ActionSample&&) = default;

		inline bool DoAction(const int time) override;

		inline void setTexture(Texture* const texture);

	private:
		Texture* _texture;
	};
} // namespace SampleBasic

extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSample* UNITY_INTERFACE_API createActionToto();

	UNITY_INTERFACE_EXPORT void setTextureActionToto(SampleBasic::ActionSample* actionToto, Texture* texture);
}
