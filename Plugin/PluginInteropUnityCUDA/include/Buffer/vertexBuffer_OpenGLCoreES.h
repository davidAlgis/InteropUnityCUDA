#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3
#include "vertexBuffer.h"

#if SUPPORT_OPENGL_UNIFIED

class VertexBuffer_OpenGLCoreES : public VertexBuffer
{
public:
	VertexBuffer_OpenGLCoreES(void* bufferHandle, int size);
	~VertexBuffer_OpenGLCoreES();
	virtual void registerBufferInCUDA();
	virtual void unRegisterBufferInCUDA();
	virtual int SetTextureFromBuffer(Texture& texture);

};

#endif