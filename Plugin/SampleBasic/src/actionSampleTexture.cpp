#pragma once
#include "ActionSampleTexture.h"
#include "unityPlugin.h"
#include "texture.h"


void kernelCallerWriteTexture(const dim3 dimGrid, const dim3 dimBlock, cudaSurfaceObject_t inputSurfaceObj, const float t, const int width, const int height);

namespace SampleBasic {

	ActionSampleTexture::ActionSampleTexture(void* texturePtr, int width, int height) : Action()
	{
		_texture = CreateTextureInterop(texturePtr, width, height, 1);
	}


	inline int ActionSampleTexture::Start()
	{
		_texture->registerTextureInCUDA();
		return 0;
	}

	int ActionSampleTexture::Update()
	{		

		cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject();
		kernelCallerWriteTexture(_texture->getDimBlock(), _texture->getDimBlock(), surf, GetTime(), _texture->getWidth(), _texture->getHeight());
		cudaDeviceSynchronize();
		_texture->unMapTextureToSurfaceObject(surf);
		return 0;
	}

	inline int ActionSampleTexture::OnDestroy()
	{
		_texture->unRegisterTextureInCUDA();
		return 0;
	}

} // namespace SampleBasic


extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTexture* UNITY_INTERFACE_API createActionSampleTextureBasic(void* texturePtr, int width, int height)
	{
		return (new SampleBasic::ActionSampleTexture(texturePtr, width, height));
	}
} // extern "C"

