#pragma once
#include "ActionSampleTextureArray.h"
#include "unityPlugin.h"
#include "texture.h"


void kernelCallerWriteTextureArray(const dim3 dimGrid, const dim3 dimBlock, cudaSurfaceObject_t inputSurfaceObj, const float time, const int width, const int height, const int depth);

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
		cudaSurfaceObject_t surf = _texture->mapTextureToSurfaceObject();
		kernelCallerWriteTextureArray(_texture->getDimBlock(), _texture->getDimBlock(), surf, GetTime(), _texture->getWidth(), _texture->getHeight(), _texture->getDepth());
		cudaDeviceSynchronize();
		_texture->unMapTextureToSurfaceObject(surf);
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

