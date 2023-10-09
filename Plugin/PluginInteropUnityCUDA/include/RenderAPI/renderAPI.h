#pragma once
#include "IUnityGraphics.h"
#include <cstddef>

/**
 * @class      RenderAPI
 *
 * @brief      Graphics abstraction of API. There are implementations of this
 * base class for D3D9, D3D11, OpenGL etc.; see individual RenderAPI_* files.
 */
class RenderAPI
{
    public:
    /**
     * @brief      Destroys the object.
     */
    virtual ~RenderAPI() = default;

    /**
     * @brief      Process general event like initialization, shutdown, device
     * loss/reset, etc.
     *
     * @param[in]  type        The type
     * @param      interfaces  The interfaces
     */
    virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                    IUnityInterfaces *interfaces) = 0;
};

/**
 * @brief      Create a graphics API implementation instance for the given API
 * type.
 *
 * @param[in]  apiType  The api type
 *
 * @return     A pointer toward the RenderAPI object (depends on the graphics
 * API)
 */
RenderAPI *CreateRenderAPI(UnityGfxRenderer apiType);
