#pragma once
#include "framework.h" 
#include "log.h"
#include "renderAPI.h"
#include "texture.h"

extern "C"
{
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API StartLog();

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int w, int h);

	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API CustomUnityPluginUnload();

	UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();
}


static bool  g_cudaRegister = false;
static RenderAPI* s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;
Texture* _currentTex = NULL;

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
