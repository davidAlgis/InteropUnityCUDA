#pragma once
#include "unity_plugin.h"
#include "factory.h"
#include "texture.h"
#include "vertex_buffer.h"
#include <assert.h>
#include <utility>

extern "C"
{

    UNITY_INTERFACE_EXPORT Texture *UNITY_INTERFACE_API
    CreateTextureInterop(void *textureHandle, int w, int h, int depth)
    {
        if (s_Graphics == NULL)
        {
            Log::log().debugLogError(
                "Unable to create texture, because Unity has not been loaded.");
            return NULL;
        }

        s_DeviceType = s_Graphics->GetRenderer();
        // create a texture in function of graphics API
        return Factory::createTexture(textureHandle, w, h, depth, s_DeviceType,
                                      s_CurrentAPI);
    }

    UNITY_INTERFACE_EXPORT VertexBuffer *UNITY_INTERFACE_API
    CreateVertexBufferInterop(void *bufferHandle, int size)
    {
        if (s_Graphics == NULL)
        {
            Log::log().debugLogError(
                "Unable to create texture, because Unity has not been loaded.");
            return NULL;
        }

        s_DeviceType = s_Graphics->GetRenderer();
        // create a buffer in function of graphics API
        return Factory::createBuffer(bufferHandle, size, s_DeviceType,
                                     s_CurrentAPI);
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTime(float time)
    {
        _time = time;
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTime()
    {
        return _time;
    }

    UNITY_INTERFACE_EXPORT bool IsSupported()
    {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0)
        {
            Log::log().debugLogError(
                "No CUDA device have been found. Interoperability could not "
                "work in this case. Use a computer with CUDA device if you "
                "want to take advantage of CUDA performance.");
            return false;
        }
        
        if(s_DeviceType == kUnityGfxRendererNull)
        {

            Log::log().debugLogError(
                "Unknown graphics API.");
            return false;
        }

        if (s_DeviceType != kUnityGfxRendererD3D11 &&
            s_DeviceType != kUnityGfxRendererOpenGLCore &&
            s_DeviceType != kUnityGfxRendererOpenGLES20 &&
            s_DeviceType != kUnityGfxRendererOpenGLES30)
        {
            Log::log().debugLogError(
                "Graphics API is not supported yet.");
            return false;
        }
        return true;
    }

    /// <summary>
    /// Initialize the correct API
    /// </summary>
    /// <param name="unityInterfaces">Unity interfaces that will be used
    /// after</param>
    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    UnityPluginLoad(IUnityInterfaces *unityInterfaces)
    {

        s_UnityInterfaces = unityInterfaces;

        s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();

        s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
    }

    /// <summary>
    /// Unregister the graphics API
    /// </summary>
    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
    {
    }

    UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API
    GetRenderEventFunc()
    {
        return OnRenderEvent;
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityShutdown()
    {
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API StartLog()
    {
        Log::log().debugLog("Initialize Log");
    }

    UNITY_INTERFACE_EXPORT size_t UNITY_INTERFACE_API
    RegisterAction(Action *action)
    {

        _registerActions.emplace_back(action);

        return _registerActions.size() - 1;
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    InitializeRegisterActions(int reserveCapacity)
    {
        _registerActions.clear();
        _registerActions.reserve(reserveCapacity);
    }
}

static void OnRenderEvent(int eventID)
{
    // Unknown / unsupported graphics device type? Do nothing
    if (s_CurrentAPI == NULL)
    {
        Log::log().debugLogError("RenderEvent stop because API is unknown.");
        return;
    }

    // there is 3 function per actions
    int realEventID = eventID / 3;

    if (realEventID >= _registerActions.size())
    {
        Log::log().debugLogError(
            "Unknown event : " + std::to_string(realEventID) +
            " has been called");
        return;
    }
    else
    {
        switch (eventID % 3)
        {
        case 0:
            _registerActions[realEventID]->Start();
            break;
        case 1:
            _registerActions[realEventID]->Update();
            break;
        case 2:
            _registerActions[realEventID]->OnDestroy();
            break;
        }
    }
}

static void OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    // Create graphics API implementation upon initialization
    if (eventType == kUnityGfxDeviceEventInitialize)
    {

        assert(s_CurrentAPI == NULL);
        s_DeviceType = s_Graphics->GetRenderer();
        s_CurrentAPI = CreateRenderAPI(s_DeviceType);
    }

    // Let the implementation process the device related events
    if (s_CurrentAPI)
    {
        s_CurrentAPI->ProcessDeviceEvent(eventType, s_UnityInterfaces);
    }

    // Cleanup graphics API implementation upon shutdown
    if (eventType == kUnityGfxDeviceEventShutdown)
    {
        delete s_CurrentAPI;
        s_CurrentAPI = NULL;
        s_DeviceType = kUnityGfxRendererNull;
        s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    }
}
