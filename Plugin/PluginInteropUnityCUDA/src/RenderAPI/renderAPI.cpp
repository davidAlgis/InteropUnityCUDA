#include "RenderAPI.h"
#include "IUnityGraphics.h"
#include "framework.h"

RenderAPI *CreateRenderAPI(UnityGfxRenderer apiType)
{
#if SUPPORT_D3D11
    if (apiType == kUnityGfxRendererD3D11)
    {
        extern RenderAPI *CreateRenderAPI_D3D11();
        return CreateRenderAPI_D3D11();
    }
#endif // if SUPPORT_D3D11

#if SUPPORT_D3D12
    if (apiType == kUnityGfxRendererD3D12)
    {
        extern RenderAPI *CreateRenderAPI_D3D12();
        return CreateRenderAPI_D3D12();
    }
#endif // if SUPPORT_D3D12

#if SUPPORT_OPENGL_UNIFIED
    if (apiType == kUnityGfxRendererOpenGLCore ||
        apiType == kUnityGfxRendererOpenGLES20 ||
        apiType == kUnityGfxRendererOpenGLES30)
    {
        extern RenderAPI *CreateRenderAPI_OpenGLCoreES(
            UnityGfxRenderer apiType);
        return CreateRenderAPI_OpenGLCoreES(apiType);
    }
#endif // if SUPPORT_OPENGL_UNIFIED

    // Unknown or unsupported graphics API
    return nullptr;
}
