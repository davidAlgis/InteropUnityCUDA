#pragma once
#include "buffer_OpenGLCoreES.h"
#include "openGLInclude.h"
#include <cuda_gl_interop.h>
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3


#if SUPPORT_OPENGL_UNIFIED

Buffer_OpenGLCoreES::Buffer_OpenGLCoreES(void* bufferHandle, int size, int stride)
	: Buffer(bufferHandle, size, stride)
{}

Buffer_OpenGLCoreES::~Buffer_OpenGLCoreES()
{
	GL_CHECK();
	CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Has to be call after the first issue plugin event 
/// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html 
/// </summary>
void Buffer_OpenGLCoreES::registerBufferInCUDA()
{
	// Update texture data, and free the memory buffer
	GLuint glBuffer = (GLuint)(size_t)(_bufferHandle);
	GL_CHECK();
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&_pGraphicsResource, glBuffer, cudaGraphicsRegisterFlagsWriteDiscard));
}


#endif // #if SUPPORT_OPENGL_UNIFIED
