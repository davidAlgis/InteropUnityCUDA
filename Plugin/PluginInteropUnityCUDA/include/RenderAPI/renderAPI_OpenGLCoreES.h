#pragma once
#include "openGL_include.h"
#include "renderAPI.h"
#include "framework.h"

// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3


#if SUPPORT_OPENGL_UNIFIED

class RenderAPI_OpenGLCoreES : public RenderAPI
{
public:
	RenderAPI_OpenGLCoreES(UnityGfxRenderer apiType);
	virtual ~RenderAPI_OpenGLCoreES();


	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces);
private:
	void CreateResources();

private:
	UnityGfxRenderer m_APIType;
};


RenderAPI* CreateRenderAPI_OpenGLCoreES(UnityGfxRenderer apiType);


#endif // #if SUPPORT_OPENGL_UNIFIED
