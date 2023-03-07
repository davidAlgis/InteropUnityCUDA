#pragma once
// Direct3D 11 implementation of Texture API
#include "texture.h"


#if SUPPORT_D3D11

#include <d3d11.h>
#include <cuda_d3d11_interop.h>
#include "IUnityGraphicsD3D11.h"
#include <assert.h>

class Texture_D3D11 : public Texture
{
public:
	Texture_D3D11(void* textureHandle, int textureWidth, int textureHeight, int textureDepth);
	~Texture_D3D11();
	virtual void registerTextureInCUDA();
	virtual void unRegisterTextureInCUDA();

};

#endif