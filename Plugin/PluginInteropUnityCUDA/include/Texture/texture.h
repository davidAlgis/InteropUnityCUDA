#pragma once
#include "log.h"
#include "cudaInclude.h"


class UNITY_INTERFACE_EXPORT Texture
{
	public:
		Texture(void* textureHandle, int textureWidth, int textureHeight);
		virtual void registerTextureInCUDA() = 0;
		virtual void unRegisterTextureInCUDA() = 0;
	
		cudaSurfaceObject_t mapTextureToSurfaceObject();
		void writeTexture(cudaSurfaceObject_t& inputSurfObj, const float time);
		void unMapTextureToSurfaceObject(cudaSurfaceObject_t& inputSurfObj);

	protected:
		void* _textureHandle;
		int _textureWidth;
		int _textureHeight;
		cudaGraphicsResource* _pGraphicsResource;

	private:
		dim3 _dimBlock;
		dim3 _dimGrid;
};
