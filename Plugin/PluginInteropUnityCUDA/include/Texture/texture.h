#pragma once
#include "log.h"
#include "cudaInclude.h"


class UNITY_INTERFACE_EXPORT Texture
{
	public:
		/// <summary>
		/// Constructor of texture
		/// </summary>
		/// <param name="textureHandle">A pointer of texture with float4 that has been generated with Unity (see function
		/// GetNativeTexturePtr https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html) </param>
		/// <param name="textureWidth">the width of the texture</param>
		/// <param name="textureHeight">the height of the texture</param>
		/// <param name="textureDepth">the depth of the texture</param>
		Texture(void* textureHandle, int textureWidth, int textureHeight, int textureDepth);


		/// <summary>
		/// Register the texture in CUDA, this has to be override because it depends on the graphics api
		/// </summary>
		virtual void registerTextureInCUDA() = 0;

		/// <summary>
		/// Unregister the texture in CUDA, this has to be override because it depends on the graphics api
		/// </summary>
		virtual void unRegisterTextureInCUDA() = 0;
	
		/// <summary>
		/// Map a cuda array to the graphics resources and wrap it into a surface object of cuda
		/// </summary>
		/// <param name="indexInArray"> Array index for array textures or cubemap face index as 
		/// defined by cudaGraphicsCubeFace for cubemap textures for the subresource to access </param>
		/// <returns>a cuda surface object on device memory and which can be edited in cuda</returns>
		cudaSurfaceObject_t mapTextureToSurfaceObject(int indexInArray = 0);

		/// <summary>
		/// Unmap the cuda array from graphics resources and destroy surface object
		/// This function will wait for all previous GPU activity to complete
		/// </summary>
		/// <param name="inputSurfObj">the surface object that has been created with <c>mapTextureToSurfaceObject</c> function</param>
		void unMapTextureToSurfaceObject(cudaSurfaceObject_t& inputSurfObj);

		/// <summary>
		/// Get the default dimension block (8,8,1)
		/// </summary>
		dim3 getDimBlock() const;

		/// <summary>
		/// Get the default dimension grid ((sizeBuffer + 7)/8,((sizeBuffer + 7)/8,1)
		/// </summary>
		dim3 getDimGrid() const;

		/// <summary>
		/// Get the width of the texture
		/// </summary>
		int getWidth() const;

		/// <summary>
		/// Get the height of the texture
		/// </summary>
		int getHeight() const;

		/// <summary>
		/// Get the depth of the texture
		/// </summary>
		int getDepth() const;

		/// <summary>
		/// Get the native texture pointer
		/// </summary>
		void* getNativeTexturePtr() const;

	protected:
		// Pointer to the texture created in Unity
		void* _textureHandle;
		// width of the texture
		int _textureWidth;
		// height of the texture
		int _textureHeight;
		// depth of the texture <2 <=> to Texture2D; >1 <=> Texture2DArray
		int _textureDepth;
		// Resource that can be used to retrieve the surface object for CUDA
		cudaGraphicsResource* _pGraphicsResource;

	private:

		dim3 _dimBlock;
		dim3 _dimGrid;
};
