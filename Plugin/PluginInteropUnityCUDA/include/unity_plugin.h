#pragma once
#include "IUnityGraphics.h"
#include "action.h"
#include "log.h"
#include "factory.h"
#include "texture.h"
#include "vertex_buffer.h"
#include "renderAPI.h"
#include <map>
#include <memory>

extern "C"
{
    // Export function for other native plugin to create interop graphics object
    
	/// <summary>
	/// Check if the device and the graphics api are supported
	/// </summary>
    UNITY_INTERFACE_EXPORT bool IsSupported();

    /// <summary>
    /// Create a texture object for interoperability from a texture pointer of
    /// unity
    /// </summary>
    /// <param name="textureHandle">A pointer of texture with float4 that has
    /// been generated with Unity (see function GetNativeTexturePtr
    /// https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html)
    /// </param> <param name="width">the width of the texture</param> <param
    /// name="height">the height of the texture</param> <param
    /// name="depth">depth of 0 or 1 are equivalent to a simple texture 2D when
    /// it greater than 1 it will be a texture 2D array </param>
    UNITY_INTERFACE_EXPORT Texture<float4> *CreateTextureInterop(void *textureHandle,
                                                         int w, int h,
                                                         int depth);

    /// <summary>
    /// Create a vertex buffer object for interoperability from a compute buffer
    /// pointer of unity
    /// </summary>
    /// <param name="bufferHandle">A pointer of computeBuffer with float4 that
    /// has been generated with Unity (see function GetNativeBufferPtr
    /// https://docs.unity3d.com/ScriptReference/ComputeBuffer.GetNativeBufferPtr.html)
    /// </param> <param name="size">the size of the computeBuffer</param>
    UNITY_INTERFACE_EXPORT VertexBuffer *CreateVertexBufferInterop(
        void *bufferHandle, int size);

    /// <summary>
    /// Set time for interoperability plugin
    /// </summary>
    /// <param name="time">new value for time</param>
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTime(float time);

    /// <summary>
    /// Get time for interoperability plugin
    /// </summary>
    /// <returns>the current time</returns>
    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTime();

    /// <summary>
    /// Return the callback which will be called from render thread
    /// </summary>
    UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API
    GetRenderEventFunc();

    /// <summary>
    /// Function to call when plugin is unloaded, indeed the function
    /// UnityUnload doesn't work see
    /// https://unity3d.atlassian.net/servicedesk/customer/portal/2/user/login?destination=portal%2F2%2FIN-13513
    /// </summary>
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityShutdown();

    /// <summary>
    /// Initialize log
    /// </summary>
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API StartLog();

    /// <summary>
    /// Register an action in plugin. When an action has been registered these
    /// functions (Start, Update, ...) can be called on render thread from Unity
    /// with GL.IssuePluginEvent
    /// https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
    /// Make sure the register has been initialized with
    /// InitializeRegisterActions function
    /// </summary>
    /// <param name="action">A pointer toward the action that we want to
    /// register</param> <returns>the eventId to recover the action that has
    /// been registered. To call these functions on render thread you must call
    /// GL.IssuePluginEvent with this Id.</returns>
    UNITY_INTERFACE_EXPORT size_t UNITY_INTERFACE_API
    RegisterAction(Action *action);

    /// <summary>
    /// Initialize the register of actions, this has to be called before
    /// register actions
    /// </summary>
    /// <param name="reserveCapacity">The default capacity for the register
    /// (just for optimization)</param>
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    InitializeRegisterActions(int reserveCapacity);
}

static float _time;

// current API used by Unity
static RenderAPI *s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;
static IUnityInterfaces *s_UnityInterfaces = NULL;
static IUnityGraphics *s_Graphics = NULL;

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
