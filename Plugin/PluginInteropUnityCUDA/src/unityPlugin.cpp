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
	void SetTextureFromUnity(void* textureHandle, int w, int h)
	{
		if (s_Graphics == NULL)
		{
			Log::log().debugLogError("Unable to create texture, because Unity has not been loaded.");
			return;
		}

		s_DeviceType = s_Graphics->GetRenderer();
		_currentTex.reset(Factory::createTexture(textureHandle, w, h, s_DeviceType));
	}

	void  SetBufferFromUnity(void* bufferHandle, int size)
	{
		if (s_Graphics == NULL)
		{
			Log::log().debugLogError("Unable to create texture, because Unity has not been loaded.");
			return;
		}

		s_DeviceType = s_Graphics->GetRenderer();
		_currentBuffer.reset(Factory::createBuffer(bufferHandle, size, s_DeviceType));
	}

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTime(float time)
	{
		_time = time;
	}


	/// <summary>
	/// Initialize the correct API
	/// </summary>
	/// <param name="unityInterfaces">Unity interfaces that will be used after</param>
	void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces* unityInterfaces)
	{
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
	UnityRenderingEvent GetRenderEventFunc()
	{
		return OnRenderEvent;
	}

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityShutdown()
	{
		if (_currentTex != NULL)
		{
			_currentTex->unRegisterTextureInCUDA();
		}

		if (_currentBuffer != NULL)
		{
			_currentBuffer->unRegisterBufferInCUDA();
		}
	}

}



int RegisterAction(Action& action)
{
	auto key = action.GetKey();
	//if the key isn't not already in map, we add it
	if (_registerActions.find(key) == _registerActions.end())
	{
		//equivalent to a insert (https://en.cppreference.com/w/cpp/container/map/operator_at)
		_registerActions[key] = action;
		return 0;
	}
	//if the key was already in map we don't add it and return -1
	return -1;	
}



static void OnRenderEvent(int eventID)
{
	// Unknown / unsupported graphics device type? Do nothing
	if (s_CurrentAPI == NULL)
	{
		Log::log().debugLogError("Unknown API.");
		return;
	}

	if (_registerActions.find(eventID) == _registerActions.end())
	{
		Log::log().debugLogError("Unknown event : " + std::to_string(eventID) + " has been called");
		return;
	}
	else
		_registerActions[eventID].DoAction();
		
	cudaSurfaceObject_t surf;
	float4* ptr;

	switch (eventID)
	{
		case 0:
			_currentTex->registerTextureInCUDA();
			break;
		
		case 1:
			surf = _currentTex->mapTextureToSurfaceObject();
			_currentTex->writeTexture(surf, _time);
			_currentTex->unMapTextureToSurfaceObject(surf);
			break;
		case 2:
			_currentBuffer->registerBufferInCUDA();
			break;
		case 3:
			ptr = _currentBuffer->mapResources();
			_currentBuffer->writeBuffer(ptr, _time);
			_currentBuffer->unmapResources();
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
		s_Graphics->UnregisterDeviceEventCallback(OnGraphicsDeviceEvent);
		
	}
}

