#pragma once
#include "framework.h"
#include "openGL_include.h"
#include "renderAPI.h"

#if SUPPORT_OPENGL_UNIFIED

/**
 * @class      RenderAPI_OpenGLCoreES
 *
 * @brief     This class describes the implementation of \ref RenderAPI for
 * OpenGL Core ES API.
 * Supports several flavors: Core, ES2, ES3
 */
class RenderAPI_OpenGLCoreES : public RenderAPI
{
    public:
    /**
     * @brief      Constructs a new instance.
     *
     * @param[in]  apiType  The API type
     */
    explicit RenderAPI_OpenGLCoreES(UnityGfxRenderer apiType);
    ~RenderAPI_OpenGLCoreES() override;

    void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                            IUnityInterfaces *interfaces) override;

    private:
    /**
     * @brief      Creates resources.
     */
    void CreateResources();

    UnityGfxRenderer _apiType;
};

/**
 * @brief      Create a graphics API implementation instance for the OpenGL Core
 * ES.
 *
 * @param[in]  apiType  The api type
 *
 * @return     A pointer toward the RenderAPI object.
 */
RenderAPI *CreateRenderAPI_OpenGLCoreES(UnityGfxRenderer apiType);

#endif // #if SUPPORT_OPENGL_UNIFIED
