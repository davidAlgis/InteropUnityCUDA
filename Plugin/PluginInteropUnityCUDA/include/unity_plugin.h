#pragma once

#include "IUnityGraphics.h"
#include "action.h"
#include "log.h"
#include "renderAPI.h"
#include <map>
#include <memory>

class Texture;
class VertexBuffer;

enum class ErrorBehavior
{
    DO_NOTHING = 1,
    ASSERT = 2,
    DISABLE_ACTION = 4,
};

extern "C"
{
    // Export function for other native plugin to create interop graphics object

    /**
     * @brief      Check if the device and the graphics api are supported
     *
     * @return     True if supported, False otherwise.
     */
    UNITY_INTERFACE_EXPORT bool IsSupported();

    /**
     * @brief      Create a texture object for interoperability from a texture
     * pointer of unity
     *
     * @param[in]  textureHandle  A pointer of texture with float4 that has
     * been generated with Unity (see function GetNativeTexturePtr
     * https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html)
     *
     * @param[in]  width  the width of the texture
     *
     * @param[in]  width  the height of the texture
     *
     * @param[in]  depth  depth of 0 or 1 are equivalent to a simple texture 2D
     * when it greater than 1 it will be a texture 2D array
     *
     * @return     The texture that has been created
     */
    UNITY_INTERFACE_EXPORT Texture *CreateTextureInterop(void *textureHandle,
                                                         int w, int h,
                                                         int depth);

    /**
     * @brief      Create a vertex buffer object for interoperability from a
     * compute buffer pointer of unity
     *
     * @param[in]  bufferHandle  A pointer of computeBuffer with float4 that
     * has been generated with Unity (see function GetNativeBufferPtr
     * https://docs.unity3d.com/ScriptReference/ComputeBuffer.GetNativeBufferPtr.html)
     *
     * @param[in]  size  the size of the computeBuffer
     *
     * @return     The vertex buffer that has been created
     */
    UNITY_INTERFACE_EXPORT VertexBuffer *CreateVertexBufferInterop(
        void *bufferHandle, int size);

    /**
     * @brief      Set time for interoperability plugin
     *
     * @param[in]  time  new value for time
     */
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTime(float time);

    /**
     * @brief      Get time for interoperability plugin
     *
     * @return     the current time set with SetTime
     */
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTime();

    /**
     * @brief      Return the callback which will be called from render thread
     *
     * @return     The callback that is called on render thread
     */
    UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API
    GetRenderEventFunc();

    /**
     * @brief      Function to call when plugin is unloaded, indeed the function
     * UnityUnload doesn't work see
     * https://unity3d.atlassian.net/servicedesk/customer/portal/2/user/login?destination=portal%2F2%2FIN-13513
     */
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityShutdown();

    /**
     * @brief      Initialize logs
     */
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API StartLog();

    /**
     * @brief      Sets the error behavior.
     *
     * @param[in]  behavior  The behavior
     */
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    SetErrorBehavior(int behavior);

    /**
     * @brief      Gets the error behavior.
     *
     * @return     The error behavior.
     */
    UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetErrorBehavior();

    /**
     * @brief      Register an action in plugin. When an action has been
     * registered these functions (Start, Update, ...) can be called on render
     * thread from Unity with GL.IssuePluginEvent
     * https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html Make
     * sure the register has been initialized with InitializeRegisterActions
     * function
     *
     * @param      action  A pointer toward the action that we want to register
     *
     * @return     The eventId to recover the action that has been registered.
     * To call these functions on render thread you must call
     * GL.IssuePluginEvent with this Id.
     */
    UNITY_INTERFACE_EXPORT size_t UNITY_INTERFACE_API
    RegisterAction(Action *action);

    /**
     * @brief      Initialize the register of actions, this has to be called
     * before register actions
     *
     * @param[in]  reserveCapacity  The default capacity for the register (just
     * for optimization)
     */
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    InitializeRegisterActions(int reserveCapacity);

    
}

static float _time;
// if it's different of 0 abort the plugin

// current API used by Unity
static RenderAPI *s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;
static IUnityInterfaces *s_UnityInterfaces = NULL;
static IUnityGraphics *s_Graphics = NULL;

static ErrorBehavior _errorBehavior = ErrorBehavior::DISABLE_ACTION;
// the register of action
static std::vector<Action *> _registerActions;

// defined the key action to use
static int _keyAction = 0;

/// <summary>
/// Callback that is called on GL.IssuePluginEvent
/// </summary>
/// <param name="eventID">define which action will be called</param>
/// <returns></returns>
static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

static void UNITY_INTERFACE_API
OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);

void DisableAction(Action* actionToDisable);
