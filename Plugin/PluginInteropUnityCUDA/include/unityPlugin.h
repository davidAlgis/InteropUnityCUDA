#pragma once
#include "framework.h" 
#include "log.h"
#include "renderAPI.h"
#include <memory>
#include <map>
#include "Action.h"

class Texture;
class VertexBuffer;


extern "C"
{
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTextureFromUnity(void* textureHandle, int w, int h);
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetBufferFromUnity(void* bufferHandle, int size);
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API SetTime(float time);

	UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API GetRenderEventFunc();
	void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityShutdown();
}



static RenderAPI* s_CurrentAPI = NULL;
static UnityGfxRenderer s_DeviceType = kUnityGfxRendererNull;
static IUnityInterfaces* s_UnityInterfaces = NULL;
static IUnityGraphics* s_Graphics = NULL;

static std::map<Action::Key, Action> _registerActions;
static int RegisterAction(const Action::Key key, Action action);
std::unique_ptr<Texture> _currentTex = NULL;
std::unique_ptr<VertexBuffer> _currentBuffer = NULL;
static float _time;

static void UNITY_INTERFACE_API OnRenderEvent(int eventID);

static void UNITY_INTERFACE_API OnGraphicsDeviceEvent(UnityGfxDeviceEventType eventType);
