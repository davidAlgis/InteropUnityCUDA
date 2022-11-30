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
		Log::log().debugLog("do ActionSample " + std::to_string(one));
		return fabs(log10_10 - one) < epsilon;
	}

	void ActionSample::setTexture(Texture* const texture)
	{
		_texture = texture;
	}

} // namespace SampleBasic


extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSample* UNITY_INTERFACE_API createActionToto()
	{
		return (new SampleBasic::ActionSample());
	}

	UNITY_INTERFACE_EXPORT void setTextureActionToto(SampleBasic::ActionSample* actionToto, Texture* texture)
	{
		actionToto->setTexture(texture);
	}
} // extern "C"
