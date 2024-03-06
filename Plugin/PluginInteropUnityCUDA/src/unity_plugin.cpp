#include "unity_plugin.h"
#include "buffer.h"
#include "factory.h"
#include "texture.h"
#include <cassert>
#include <utility>

extern "C"
{

    UNITY_INTERFACE_EXPORT Texture *UNITY_INTERFACE_API
    CreateTextureInterop(void *textureHandle, int w, int h, int depth)
    {
        if (sGraphics == nullptr)
        {
            Log::log().debugLogError(
                "Unable to create texture, because Unity has not been loaded.");
            return nullptr;
        }

        sDeviceType = sGraphics->GetRenderer();
        // create a texture in function of graphics API
        return Factory::createTexture(textureHandle, w, h, depth, sDeviceType,
                                      sCurrentApi);
    }

    UNITY_INTERFACE_EXPORT Buffer *UNITY_INTERFACE_API
    CreateBufferInterop(void *bufferHandle, int size)
    {
        if (sGraphics == nullptr)
        {
            Log::log().debugLogError(
                "Unable to create texture, because Unity has not been loaded.");
            return nullptr;
        }

        sDeviceType = sGraphics->GetRenderer();
        // create a buffer in function of graphics API
        return Factory::createBuffer(bufferHandle, size, sDeviceType,
                                     sCurrentApi);
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTimeInterop(float time)
    {
        _time = time;
    }

    UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTimeInterop()
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

        sDeviceType = sGraphics->GetRenderer();

        if (sDeviceType == kUnityGfxRendererNull)
        {

            Log::log().debugLogError("Unknown graphics API.");
            return false;
        }

        if (sDeviceType != kUnityGfxRendererD3D11 &&
            sDeviceType != kUnityGfxRendererOpenGLCore &&
            sDeviceType != kUnityGfxRendererOpenGLES20 &&
            sDeviceType != kUnityGfxRendererOpenGLES30)
        {
            Log::log().debugLogError(("Graphics API " +
                                      std::to_string(sDeviceType) +
                                      " is not supported yet.")
                                         .c_str());
            return false;
        }
        return true;
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    UnityPluginLoad(IUnityInterfaces *unityInterfaces)
    {

        sUnityInterfaces = unityInterfaces;

        sGraphics = sUnityInterfaces->Get<IUnityGraphics>();

        sGraphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

        OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
    }

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

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    SetErrorBehavior(int behavior)
    {
        auto errorBehavior = static_cast<ErrorBehavior>(behavior);
        errorBehavior = errorBehavior;
    }

    UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API GetErrorBehavior()
    {
        return static_cast<int>(errorBehavior);
    }

    UNITY_INTERFACE_EXPORT size_t UNITY_INTERFACE_API
    RegisterAction(Action *action)
    {
        action->IsActive = true;
        registerActions.emplace_back(action);

        return registerActions.size() - 1;
    }

    UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
    InitializeRegisterActions(int reserveCapacity)
    {
        registerActions.clear();
        registerActions.reserve(reserveCapacity);
    }
}

void DisableAction(Action *actionToDisable)
{
    actionToDisable->IsActive = false;
}

static void OnRenderEvent(int eventID)
{
    // Unknown / unsupported graphics device type? Do nothing
    if (sCurrentApi == nullptr)
    {
        Log::log().debugLogError("RenderEvent stop because API is unknown.");
        return;
    }

    // there is 3 function per actions
    int realEventID = eventID / 3;

    if (realEventID >= registerActions.size())
    {
        Log::log().debugLogError(
            ("Unknown event : " + std::to_string(realEventID) +
             " has been called")
                .c_str());
        return;
    }
    else
    {
        if (registerActions[realEventID]->IsActive == false)
        {
            return;
        }

        int ret;
        switch (eventID % 3)
        {
        case 0:
            ret = registerActions[realEventID]->Start();
            break;
        case 1:
            ret = registerActions[realEventID]->Update();
            break;
        case 2:
            ret = registerActions[realEventID]->OnDestroy();
            break;
        }

        switch (errorBehavior)
        {

        case ErrorBehavior::DO_NOTHING:
            break;
        case ErrorBehavior::ASSERT:
            assert(ret == 0);
            break;
        case ErrorBehavior::DISABLE_ACTION:
            if (ret != 0)
            {
                DisableAction(registerActions[realEventID]);
                Log::log().debugLogError(
                    ("There has been an error with action " +
                     std::to_string(realEventID) + ". It has been disabled.")
                        .c_str());
            }
            break;
        default:
            break;
        }
    }
}

static void OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType)
{
    // Create graphics API implementation upon initialization
    if (eventType == kUnityGfxDeviceEventInitialize)
    {

        assert(sCurrentApi == NULL);
        sDeviceType = sGraphics->GetRenderer();
        sCurrentApi = CreateRenderAPI(sDeviceType);
    }

    // Let the implementation process the device related events
    if (sCurrentApi)
    {
        sCurrentApi->ProcessDeviceEvent(eventType, sUnityInterfaces);
    }

    // Cleanup graphics API implementation upon shutdown
    if (eventType == kUnityGfxDeviceEventShutdown)
    {
        delete sCurrentApi;
        sCurrentApi = nullptr;
        sDeviceType = kUnityGfxRendererNull;
        sGraphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
    }
}
