#pragma once
#include "ActionSampleTextureArray.h"
#include "unityPlugin.h"
#include "texture.h"

void kernelCallerWriteTexture(const dim3 dimGrid, const dim3 dimBlock, cudaSurfaceObject_t inputSurfaceObj, const float t, const int width, const int height);


namespace SampleBasic {

	ActionSampleTextureArray::ActionSampleTextureArray(void* texturePtr, int width, int height, int depth) : Action()
	{
		_texture = CreateTextureInterop(texturePtr, width, height, depth);
	}


	inline int ActionSampleTextureArray::Start()
	{
		_texture->registerTextureInCUDA();
		return 0;
	}

	int ActionSampleTextureArray::Update()
	{		
		for (int i = 0; i < _texture->getDepth(); i++)
		{
			cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject(i);
			kernelCallerWriteTexture(_texture->getDimGrid(), _texture->getDimBlock(), surf, GetTime()+2*i, _texture->getWidth(), _texture->getHeight());
			_texture->unMapTextureToSurfaceObject(surf);
		}

		cudaDeviceSynchronize();
		return 0;
	}

	inline int ActionSampleTextureArray::OnDestroy()
	{
		_texture->unRegisterTextureInCUDA();
		return 0;
	}

} // namespace SampleBasic


extern "C" {

	UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTextureArray* UNITY_INTERFACE_API createActionSampleTextureArrayBasic(void* texturePtr, int width, int height, int depth)
	{
		return (new SampleBasic::ActionSampleTextureArray(texturePtr, width, height, depth));
	}
} // extern "C"

