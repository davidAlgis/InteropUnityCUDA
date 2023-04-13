#pragma once
#include "vertex_buffer_OpenGLCoreES.h"

#if SUPPORT_OPENGL_UNIFIED

VertexBuffer_OpenGLCoreES::VertexBuffer_OpenGLCoreES(void* bufferHandle, int size)
	: VertexBuffer(bufferHandle, size)
{}

VertexBuffer_OpenGLCoreES::~VertexBuffer_OpenGLCoreES()
{
	// final check to be sure there was no mistake
	GL_CHECK();
	CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Register the buffer from OpenGL to CUDA
/// </summary>
int VertexBuffer_OpenGLCoreES::registerBufferInCUDA()
{
	// cast the pointer on the buffer of unity to gluint 
	// (first to size_t to avoid a warning C4311)
	GLuint glBuffer = (GLuint)(size_t)(_bufferHandle);
	//register the buffer to cuda : it initialize the _pGraphicsResource
	CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(&_graphicsResource, glBuffer, cudaGraphicsRegisterFlagsNone));
	return SUCCESS_INTEROP_CODE;
}

int VertexBuffer_OpenGLCoreES::unregisterBufferInCUDA()
{
	CUDA_CHECK_RETURN(cudaGraphicsUnregisterResource(_graphicsResource));
	return SUCCESS_INTEROP_CODE;
}


#endif // #if SUPPORT_OPENGL_UNIFIED
