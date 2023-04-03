#pragma once
#include "action_sample_texture.h"
#include "unity_plugin.h"

namespace SampleBasic
{

ActionSampleTexture::ActionSampleTexture(void *texturePtr, int width,
                                         int height)
    : Action()
{
    _texture = CreateTextureInterop(texturePtr, width, height, 1);
}

inline int ActionSampleTexture::Start()
{
    _texture->registerTextureInCUDA();
    _texture->mapTextureToSurfaceObject();
    return 0;
}

int ActionSampleTexture::Update()
{
    kernelCallerWriteTexture(_texture->getDimGrid(), _texture->getDimBlock(),
                             _texture->getSurfaceObject(), GetTime(), _texture->getWidth(),
                             _texture->getHeight());
    // cudaDeviceSynchronize();
    return 0;
}

inline int ActionSampleTexture::OnDestroy()
{
    _texture->unmapTextureToSurfaceObject();
    _texture->unregisterTextureInCUDA();
    return 0;
}

} // namespace SampleBasic

extern "C"
{

    UNITY_INTERFACE_EXPORT SampleBasic::ActionSampleTexture *UNITY_INTERFACE_API
    createActionSampleTextureBasic(void *texturePtr, int width, int height)
    {
        return (
            new SampleBasic::ActionSampleTexture(texturePtr, width, height));
    }
} // extern "C"
