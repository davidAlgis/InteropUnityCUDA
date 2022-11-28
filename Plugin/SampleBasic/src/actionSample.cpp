#pragma once
#include "actionSample.h"
#include "action.h"
#include <memory>



namespace SampleBasic {

	bool ActionSample::DoAction(const int time)
	{
		constexpr auto epsilon = 1e6;

		const auto log10_10 = log10(10.0);
		const auto one = 1.0 + 0.0 * time;

		return fabs(log10_10 - one) < epsilon;
	}

	void ActionSample::setTexture(Texture* const texture)
	{
		_texture = texture;
	}

} // namespace SampleBasic


extern "C" {

	SampleBasic::ActionSample* UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API createActionToto(Action::Key key)
	{
		return (new SampleBasic::ActionSample(key));
	}

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
		setTextureActionToto(SampleBasic::ActionSample* actionToto, Texture* texture)
	{
		actionToto->setTexture(texture);
	}
} // extern "C"
