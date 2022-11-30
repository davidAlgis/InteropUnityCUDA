#pragma once
#include "unityPlugin.h"
#include "factory.h"
#include "texture.h"
#include "vertexBuffer.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <utility>

extern "C"
{
	/// <summary>
	/// SetTextureFromUnity, an example function we export which is called by one of the scripts.
	/// </summary>
	/// <param name="textureHandle"></param>
	/// <param name="w"></param>
	/// <param name="h"></param>
	UNITY_INTERFACE_EXPORT Texture* UNITY_INTERFACE_API CreateTextureInterop(void* textureHandle, int w, int h)
	{
		if (s_Graphics == NULL)
		{
			Log::log().debugLogError("Unable to create texture, because Unity has not been loaded.");
			return NULL;
		}

		s_DeviceType = s_Graphics->GetRenderer();
		return Factory::createTexture(textureHandle, w, h, s_DeviceType);
	}

	UNITY_INTERFACE_EXPORT VertexBuffer* UNITY_INTERFACE_API CreateVertexBufferInterop(void* bufferHandle, int size)
	{
		if (s_Graphics == NULL)
		{
			Log::log().debugLogError("Unable to create texture, because Unity has not been loaded.");
			return NULL;
		}

		s_DeviceType = s_Graphics->GetRenderer();
		return Factory::createBuffer(bufferHandle, size, s_DeviceType);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API SetTime(float time)
	{
		_time = time;
	}

	UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTime()
	{
		return _time;
	}


	/// <summary>
	/// Initialize the correct API
	/// </summary>
	/// <param name="unityInterfaces">Unity interfaces that will be used after</param>
	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
	{
		Log::log().debugLog("load");
		//_registerActions.reserve(16);
		s_UnityInterfaces = unityInterfaces;

		s_Graphics = s_UnityInterfaces->Get<IUnityGraphics>();		

		s_Graphics->RegisterDeviceEventCallback(OnGraphicsDeviceEvent);

#if SUPPORT_VULKAN
		if (s_Graphics->GetRenderer() == kUnityGfxRendererNull)
		{
			extern void RenderAPI_Vulkan_OnPluginLoad(IUnityInterfaces*);
			RenderAPI_Vulkan_OnPluginLoad(unityInterfaces);
		}
#endif 

		OnGraphicsDeviceEvent(kUnityGfxDeviceEventInitialize);
	}

	

	/// <summary>
	/// Unregister the graphics API
	/// </summary>
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
	{
	}

	/// <summary>
	/// GetRenderEventFunc, an example function we export which is used to get a rendering event callback function.
	/// </summary>
	UNITY_INTERFACE_EXPORT UnityRenderingEvent UNITY_INTERFACE_API GetRenderEventFunc()
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

	UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API RegisterAction(Action* action)
	{

		_registerActions.emplace_back(action);

		return _registerActions.size() - 1;
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API InitializeRegisterActions(int reserveCapacity)
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
		Log::log().debugLogError("Unknown API.");
		return;
	}

	int realEventID = eventID / 3;

	if (realEventID >= _registerActions.size())
	{
		Log::log().debugLogError("Unknown event : " + std::to_string(realEventID) + " has been called");
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

	//cudaSurfaceObject_t surf;
	//float4* ptr;

	//switch (eventID)
	//{
	//	case 0:
	//		_currentTex->registerTextureInCUDA();
	//		break;
	//	
	//	case 1:
	//		surf = _currentTex->mapTextureToSurfaceObject();
	//		_currentTex->writeTexture(surf, _time);
	//		_currentTex->unMapTextureToSurfaceObject(surf);
	//		break;
	//	case 2:
	//		_currentBuffer->registerBufferInCUDA();
	//		break;
	//	case 3:
	//		ptr = _currentBuffer->mapResources();
	//		_currentBuffer->writeBuffer(ptr, _time);
	//		_currentBuffer->unmapResources();
	//		break;
	//	
	//}

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

