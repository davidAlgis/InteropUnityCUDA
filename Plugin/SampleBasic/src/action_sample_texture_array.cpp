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
}

inline int ActionSampleTextureArray::Start()
{
    int ret = _texture->registerTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the registration of "
                 "the texture in CUDA. Abort ActionSampleTextureArray !");
    ret = _texture->mapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the map of "
                 "the texture to surface object in CUDA. Abort "
                 "ActionSampleTextureArray !");
    return 0;
}

int ActionSampleTextureArray::Update()
{
    kernelCallerWriteTextureArray(
        _texture->getDimGrid(), _texture->getDimBlock(),
        _texture->getSurfaceObjectArray(), GetTimeInterop(),
        _texture->getWidth(), _texture->getHeight(), _texture->getDepth());
    cudaDeviceSynchronize();
    int ret = CUDA_CHECK(cudaGetLastError());
    GRUMBLE(ret, "There has been an error during the update. "
                 "Abort ActionSampleTextureArray !");
    return 0;
}

inline int ActionSampleTextureArray::OnDestroy()
{
    int ret = _texture->unmapTextureToSurfaceObject();
    GRUMBLE(ret, "There has been an error during the unmap of "
                 "the texture to surface object in CUDA. Abort "
                 "ActionSampleTextureArray !");
    ret = _texture->unregisterTextureInCUDA();
    GRUMBLE(ret, "There has been an error during the unregistration of "
                 "the texture in CUDA. Abort ActionSampleTextureArray !");
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
