#include "renderAPI_OpenGLCoreES.h"

#if SUPPORT_OPENGL_UNIFIED

RenderAPI *CreateRenderAPI_OpenGLCoreES(UnityGfxRenderer apiType)
{
    return new RenderAPI_OpenGLCoreES(apiType);
}

void RenderAPI_OpenGLCoreES::CreateResources()
{
    gl3wInit();
    GL_CHECK();
}

RenderAPI_OpenGLCoreES::RenderAPI_OpenGLCoreES(UnityGfxRenderer apiType)
    : _apiType(apiType)
{
}

RenderAPI_OpenGLCoreES::~RenderAPI_OpenGLCoreES()
{
}

void RenderAPI_OpenGLCoreES::ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                                IUnityInterfaces *interfaces)
{
    if (type == kUnityGfxDeviceEventInitialize)
    {
        CreateResources();
    }
    else if (type == kUnityGfxDeviceEventShutdown)
    {
        GL_CHECK();
    }
}

#endif // #if SUPPORT_OPENGL_UNIFIED
