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
    _surf = new cudaSurfaceObject_t[depth];
    for(int i=0; i<depth; i++)
    {
        _surf[i] = 0;
    }
}

inline int ActionSampleTextureArray::Start()
{
    _texture->registerTextureInCUDA();
    _surf = _texture->mapTextureArrayToSurfaceObject();
    return 0;
}

int ActionSampleTextureArray::Update()
{
    kernelCallerWriteTextureArray(
        _texture->getDimGrid(), _texture->getDimBlock(),
                             _surf[0], GetTime(), _texture->getWidth(),
                             _texture->getHeight(), _texture->getDepth());
    kernelCallerWriteTextureArray(
        _texture->getDimGrid(), _texture->getDimBlock(),
                             _surf[1], 2*GetTime(), _texture->getWidth(),
                             _texture->getHeight(), _texture->getDepth());

    cudaDeviceSynchronize();
    return 0;
}

inline int ActionSampleTextureArray::OnDestroy()
{
    _texture->unMapTextureToSurfaceObject(_surf[0]);
    _texture->unRegisterTextureInCUDA();
    delete(_surf);
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
