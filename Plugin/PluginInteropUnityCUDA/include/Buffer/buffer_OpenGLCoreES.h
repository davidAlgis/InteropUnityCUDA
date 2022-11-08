#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3
#include "buffer.h"

#if SUPPORT_OPENGL_UNIFIED

class Buffer_OpenGLCoreES : public Buffer
{
public:
	Buffer_OpenGLCoreES(void* bufferHandle, int size, int stride);
	~Buffer_OpenGLCoreES();
	virtual void registerBufferInCUDA();

};

#endif