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
    int ret = _texture->registerTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
            "the texture in CUDA. Abort ActionSampleTexture !");
    ret = _texture->mapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the map of "
                                 "the texture to surface object in CUDA. Abort "
                                 "ActionSampleTexture !");
    return 0;
}

int ActionSampleTexture::Update()
{
    kernelCallerWriteTexture(_texture->getDimGrid(), _texture->getDimBlock(),
                             _texture->getSurfaceObject(), GetTime(),
                             _texture->getWidth(), _texture->getHeight());
    cudaDeviceSynchronize();
    int ret = CUDA_CHECK(cudaGetLastError());
    GRUMBLE(ret, "There has been an error during the update. "
                                 "Abort ActionSampleTexture !");
    return 0;
}

inline int ActionSampleTexture::OnDestroy()
{
    int ret = _texture->unmapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the unmap of "
                                 "the texture to surface object in CUDA. Abort "
                                 "ActionSampleTexture !");
    ret = _texture->unregisterTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
            "the texture in CUDA. Abort ActionSampleTexture !");
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
