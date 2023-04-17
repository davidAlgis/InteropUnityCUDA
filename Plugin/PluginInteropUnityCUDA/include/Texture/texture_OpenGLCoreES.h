#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of Textures API.
// Supports several flavors: Core, ES2, ES3
#include "texture.h"


#if SUPPORT_OPENGL_UNIFIED

#include "openGL_include.h"
#include <cuda_gl_interop.h>


class Texture_OpenGLCoreES : public Texture
{
public:
	Texture_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight, int textureDepth);
	~Texture_OpenGLCoreES();
	virtual int registerTextureInCUDA();
	virtual int unregisterTextureInCUDA();
	virtual int generateMips();

protected:
	virtual int copyUnityTextureToAPITexture();
	virtual int copyAPITextureToUnityTexture();
private:
	GLuint _texUnityGL;
	GLenum _texTarget;
};

#endif