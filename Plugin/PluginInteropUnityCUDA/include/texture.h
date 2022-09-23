#pragma once
#include "log.h"
#include "cuda_runtime.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char* file, int line);


class Texture
{
	public:
		Texture(void* textureHandle, int textureWidth, int textureHeight);
		virtual void registerTextureInCUDA() = 0;
	
		cudaSurfaceObject_t mapTextureToSurfaceObject();
		void writeTexture(cudaSurfaceObject_t& inputSurfObj);
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



// Create a graphics API implementation instance for the given API type.
Texture* createTextureAPI(void* textureHandle, int textureWidth, int textureHeight, UnityGfxRenderer apiType);

