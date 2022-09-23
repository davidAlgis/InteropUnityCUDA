#pragma once
#include "texture.h"
#include "openGLInclude.h"
#include <cuda_gl_interop.h>
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3


#if SUPPORT_OPENGL_UNIFIED



class Texture_OpenGLCoreES : public Texture
{
public:
	Texture_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight);
	~Texture_OpenGLCoreES();
	virtual void registerTextureInCUDA();

};

Texture_OpenGLCoreES::Texture_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight) 
	: Texture(textureHandle, textureWidth, textureHeight)
{
	Log::log().debugLog("Texture_OpenGLCoreES constructor");
	registerTextureInCUDA();
}


Texture_OpenGLCoreES::~Texture_OpenGLCoreES()
{
	GL_CHECK();
	CUDA_CHECK(cudaGetLastError());
};

void Texture_OpenGLCoreES::registerTextureInCUDA()
{
	// Update texture data, and free the memory buffer
	Log::log().debugLog(std::to_string((int)_textureHandle));
	GLuint gltex = (GLuint)(size_t)(_textureHandle);
	glBindTexture(GL_TEXTURE_2D, gltex);
	GL_CHECK();
	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&_pGraphicsResource, gltex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}


Texture* createTextureAPI_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight)
{
	return new Texture_OpenGLCoreES(textureHandle, textureWidth, textureHeight);
}



#endif // #if SUPPORT_OPENGL_UNIFIED
