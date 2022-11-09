#pragma once
#include "vertexBuffer_OpenGLCoreES.h"
#include "openGLInclude.h"
#include <cuda_gl_interop.h>
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3


#if SUPPORT_OPENGL_UNIFIED

VertexBuffer_OpenGLCoreES::VertexBuffer_OpenGLCoreES(void* bufferHandle, int size)
	: VertexBuffer(bufferHandle, size)
{}

VertexBuffer_OpenGLCoreES::~VertexBuffer_OpenGLCoreES()
{
	GL_CHECK();
	CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Has to be call after the first issue plugin event 
/// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html 
/// </summary>
void VertexBuffer_OpenGLCoreES::registerBufferInCUDA()
{
	// Update texture data, and free the memory buffer
	GLuint glBuffer = (GLuint)(_bufferHandle);
	GL_CHECK();
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&_pGraphicsResource, glBuffer, cudaGraphicsRegisterFlagsNone));
}

void VertexBuffer_OpenGLCoreES::unRegisterBufferInCUDA()
{
	CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
}


#endif // #if SUPPORT_OPENGL_UNIFIED
