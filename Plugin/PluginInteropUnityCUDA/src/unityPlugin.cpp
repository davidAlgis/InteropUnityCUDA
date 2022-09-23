#pragma once
#include "unityPlugin.h"
#include <iostream>
#include <assert.h>
#include <math.h>


extern "C"
{
    void StartLog()
    {
        Log::log().debugLog("");
    }


	/// <summary>
	/// SetTextureFromUnity, an example function we export which is called by one of the scripts.
	/// </summary>
	/// <param name="textureHandle"></param>
	/// <param name="w"></param>
	/// <param name="h"></param>
	void SetTextureFromUnity(void* textureHandle, int w, int h)
	{
		Log::log().debugLog(std::to_string((int)textureHandle));
		if (s_Graphics == NULL)
		{
			Log::log().debugLogError("Unable to create texture, because Unity has not been loaded.");
			return;
		}

		s_DeviceType = s_Graphics->GetRenderer();
		_currentTex = createTextureAPI(textureHandle, w, h, s_DeviceType);
		g_cudaRegister = false;
	}


	/// <summary>
	/// Initialize the correct API
	/// </summary>
	/// <param name="unityInterfaces">Unity interfaces that will be used after</param>
	void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
	{
		Log::log().debugLog("Unity plugin load");
		s_UnityInterfaces = unityInterfaces;

		s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();		

		s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

#if SUPPORT_VULKAN
		if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
		{
			extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
			RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
		}
#endif // SUPPORT_VULKAN

		// Run OnGraphicsDeviceEvent(initialize) manually on plugin load
		OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
	}


	void CustomUnityPluginUnload()
	{
		Log::log().debugLog("unload");
	}
	

	/// <summary>
	/// Unregister the graphics API
	/// </summary>
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
	{
		CustomUnityPluginUnload();
	}

	/// <summary>
	/// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.
	/// </summary>
	UnityRenderingEvent GetRenderEventFunc()
	{
		return OnRenderEvent;
	}

}



static void OnRenderEvent(int eventID)
{
	// Unknown / unsupported graphics device type? Do nothing
	if (s_CurrentAPI == NULL)
	{
		Log::log().debugLogError("Unknown API.");
		return;
	}


	switch (eventID)
	{
		case 0:
			_currentTex->registerTextureInCUDA();
			break;
		case 1:
			auto surf = _currentTex->mapTextureToSurfaceObject();
			_currentTex->writeTexture(surf);
			_currentTex->unMapTextureToSurfaceObject(surf);
			break;
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
		Log::log().debugLog("kUnityGfxDeviceEventShutdown");

		s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
	}
}

