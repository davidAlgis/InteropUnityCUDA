#pragma once
#include "actionSample.h"
#include "action.h"
#include "unityPlugin.h"
#include <memory>


namespace SampleBasic {

	ActionSample::ActionSample(void* texturePtr, int width, int height) : Action()
	{
		_texture = CreateTextureInterop(texturePtr, width, height);
		_hasBeenRegistered = false;
	}

	ActionSample::~ActionSample()
	{
		_texture->unRegisterTextureInCUDA();
	}

	inline int ActionSample::Start()
	{
		_texture->registerTextureInCUDA();
		return 0;
	}

	int ActionSample::Update()
	{		
		cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject();
		_texture->writeTexture(surf, GetTime());
		_texture->unMapTextureToSurfaceObject(surf);
		return 0;
	}

	inline int ActionSample::OnDestroy()
	{
		return 0;
	}


} // namespace SampleBasic


extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSample* UNITY_INTERFACE_API createActionSampleBasic(void* texturePtr, int width, int height)
	{
		return (new SampleBasic::ActionSample(texturePtr, width, height));
	}
} // extern "C"
