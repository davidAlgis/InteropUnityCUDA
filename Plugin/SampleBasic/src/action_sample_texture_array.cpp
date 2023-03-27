#pragma once
#include "action_sample_texture_array.h"
#include "unity_plugin.h"

namespace SampleBasic
{

ActionSampleTextureArray::ActionSampleTextureArray(void *texturePtr, int width,
                                                   int height, int depth)
    : Action()
{
    _texture = CreateTextureInterop(texturePtr, width, height, depth);
    _surf = 0;
}

inline int ActionSampleTextureArray::Start()
{
    _texture->registerTextureInCUDA();
    _surf = _texture->mapTextureToSurfaceObject(1);
    return 0;
}

int ActionSampleTextureArray::Update()
{
    kernelCallerWriteTextureArray(
        _texture->getDimGrid(), _texture->getDimBlock(),
                             _surf, GetTime(), _texture->getWidth(),
                             _texture->getHeight(), _texture->getDepth());

    cudaDeviceSynchronize();
    return 0;
}

inline int ActionSampleTextureArray::OnDestroy()
{
    _texture->unMapTextureToSurfaceObject(_surf);
    _texture->unRegisterTextureInCUDA();
    return 0;
}

} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTextureArray
        *UNITY_INTERFACE_API
        createActionSampleTextureArrayBasic(void *texturePtr, int width,
                                            int height, int depth)
    {
        return (new SampleBasic::ActionSampleTextureArray(texturePtr, width,
                                                          height, depth));
    }
} // extern "C"
