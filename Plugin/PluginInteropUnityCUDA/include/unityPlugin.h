#pragma once
#include "framework.h" 
#include "log.h"
#include "renderAPI.h"
#include "action.h"
#include <memory>
#include <map>

class Texture;
class VertexBuffer;


extern "C"
{
	UNITY_INTERFACE_EXPORT Texture* CreateTextureInterop(void* textureHandle, int w, int h);
	UNITY_INTERFACE_EXPORT VertexBuffer*  CreateVertexBufferInterop(void* bufferHandle, int size);

	UNITY_INTERFACE_EXPORT void  UNITY_INTERFACE_API SetTime(float time);
	UNITY_INTERFACE_EXPORT float UNITY_INTERFACE_API GetTime();

	UNITY_INTERFACE_EXPORT UnityRenderingEvent  UNITY_INTERFACE_API GetRenderEventFunc();
	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityShutdown();

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API StartLog();

	UNITY_INTERFACE_EXPORT int UNITY_INTERFACE_API RegisterAction(Action* action);
	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API InitializeRegisterActions(int reserveCapacity);
}



static float _time;

static RenderAPI* s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;

static std::vector<Action*> _registerActions;

static int _keyAction = 0;

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
