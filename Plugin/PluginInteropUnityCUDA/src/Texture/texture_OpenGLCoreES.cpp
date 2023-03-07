#pragma once
#include "texture_OpenGLCoreES.h"

#if SUPPORT_OPENGL_UNIFIED


Texture_OpenGLCoreES::Texture_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight, int textureDepth)
	: Texture(textureHandle, textureWidth, textureHeight, textureDepth)
{}

Texture_OpenGLCoreES::~Texture_OpenGLCoreES()
{
	GL_CHECK();
	CUDA_CHECK(cudaGetLastError());
};

/// <summary>
/// Has to be call after the first issue plugin event 
/// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html 
/// register a graphics resources defined from the texture openGL
/// </summary>
void Texture_OpenGLCoreES::registerTextureInCUDA()
{
	// if depth is < 2 it's a texture2D, else it's a texture2DArray
	GLenum target = _textureDepth < 2 ? GL_TEXTURE_2D : GL_TEXTURE_3D;

	// cast the pointer on the texture of unity to gluint
	GLuint gltex = (GLuint)(size_t)(_textureHandle);
	//glBindTexture(target, gltex);
	GL_CHECK();
	// register the texture to cuda : it initialize the _pGraphicsResource
	CUDA_CHECK(cudaGraphicsGLRegisterImage(&_pGraphicsResource, gltex, target, cudaGraphicsRegisterFlagsWriteDiscard));
}


void Texture_OpenGLCoreES::unRegisterTextureInCUDA()
{
	CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
}

#endif // #if SUPPORT_OPENGL_UNIFIED
