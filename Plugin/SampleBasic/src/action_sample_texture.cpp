#pragma once
#include "action_sample_texture.h"
#include "texture.h"
#include "unity_plugin.h"

void kernelCallerWriteTexture(const dim3 dimGrid, const dim3 dimBlock,
                              cudaSurfaceObject_t inputSurfaceObj,
                              const float t, const int width, const int height);

namespace SampleBasic
{

ActionSampleTexture::ActionSampleTexture(void *texturePtr, int width,
                                         int height)
    : Action()
{
    _texture = CreateTextureInterop(texturePtr, width, height, 1);
    _surf = 0;
}

inline int ActionSampleTexture::Start()
{
    _texture->registerTextureInCUDA();
    _surf = _texture->mapTextureToSurfaceObject();
    return 0;
}

int ActionSampleTexture::Update()
{
    // _texture->copyUnityTextureToAPITexture();
    kernelCallerWriteTexture(_texture->getDimGrid(), _texture->getDimBlock(),
                             _surf, GetTime(), _texture->getWidth(),
                             _texture->getHeight());
    // cudaDeviceSynchronize();
    return 0;
}

inline int ActionSampleTexture::OnDestroy()
{
    _texture->unMapTextureToSurfaceObject(_surf);
    _texture->unRegisterTextureInCUDA();
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
