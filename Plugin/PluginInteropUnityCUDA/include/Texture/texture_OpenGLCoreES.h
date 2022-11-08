#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of RenderAPI.
// Supports several flavors: Core, ES2, ES3
#include "texture.h"


#if SUPPORT_OPENGL_UNIFIED


class Texture_OpenGLCoreES : public Texture
{
public:
	Texture_OpenGLCoreES(void* textureHandle, int textureWidth, int textureHeight);
	~Texture_OpenGLCoreES();
	virtual void registerTextureInCUDA();

};

#endif