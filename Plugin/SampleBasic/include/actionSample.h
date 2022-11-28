#pragma once
#include "action.h"
#include "log.h"
#include "texture.h"

namespace SampleBasic {
	class ActionSample : public Action {
	public:
		ActionSample(Action::Key key) : Action(key) {}
		ActionSample(const ActionSample&) = default;
		ActionSample(ActionSample&&) = default;

		// TODO: verify this
		~ActionSample() = default;
		ActionSample& operator=(const ActionSample&) = default;
		ActionSample& operator=(ActionSample&&) = default;

		inline bool DoAction(const int time) override;

		inline void setTexture(Texture* const texture);

	private:
		Texture* _texture;
	};
} // namespace SampleBasic

extern "C" {

	SampleBasic::ActionSample* UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API createActionToto(int key);

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API setTextureActionToto(SampleBasic::ActionSample*,
		Texture*);
}
