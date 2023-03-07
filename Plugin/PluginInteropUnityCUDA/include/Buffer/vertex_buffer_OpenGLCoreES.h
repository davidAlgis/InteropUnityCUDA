#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of Vertex Buffer.
// Supports several flavors: Core, ES2, ES3
#include "vertex_buffer.h"

#if SUPPORT_OPENGL_UNIFIED

#include "openGL_include.h"
#include <cuda_gl_interop.h>

class VertexBuffer_OpenGLCoreES : public VertexBuffer
{
public:
	VertexBuffer_OpenGLCoreES(void* bufferHandle, int size);
	~VertexBuffer_OpenGLCoreES();
	virtual void registerBufferInCUDA();
	virtual void unRegisterBufferInCUDA();

};

#endif
