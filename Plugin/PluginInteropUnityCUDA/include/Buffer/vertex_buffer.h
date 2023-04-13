#pragma once
#include "texture.h"
#include "IUnityGraphics.h"
#include "cuda_include.h"
#include "log.h"

class VertexBuffer
{
	public:

		/**
		 * Constructor of vertex buffer
		 * @param  bufferHandle A pointer of computeBuffer with float4 that has been generated with Unity (see function
		 * GetNativeBufferPtr https://docs.unity3d.com/ScriptReference/ComputeBuffer.GetNativeBufferPtr.html)
		 * @param  size         the size of the computeBuffer
		 */
		UNITY_INTERFACE_EXPORT VertexBuffer(void* bufferHandle, int size);

		/**
		 * Register the buffer in CUDA, this has to be override because it depends on the graphics api
		 */
		UNITY_INTERFACE_EXPORT virtual int registerBufferInCUDA() = 0;

		/**
		 * Unregister the buffer in CUDA, this has to be override because it depends on the graphics api
		 */
		UNITY_INTERFACE_EXPORT virtual int unregisterBufferInCUDA() = 0;


		/**
		 * Map resources to CUDA
		 * return an array of T* defined on device memory and which can be edited in cuda
		 */
		template <typename T>
		UNITY_INTERFACE_EXPORT int mapResources(T** vertexPtr)
		{
			// map resource
			CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &_graphicsResource, 0));
			// pointer toward an array of float4 on device memory : the compute buffer
			// number of bytes that has been readed
			size_t numBytes;
			// map the resources on a float4 array that can be modify on device
			CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)vertexPtr, &numBytes,
				_graphicsResource));
			return SUCCESS_INTEROP_CODE;
		}

		/**
		 * Unmap resources from CUDA
		 * This function will wait for all previous GPU activity to complete
		 */
		UNITY_INTERFACE_EXPORT int unmapResources();

		/**
		 * Get the default dimension block (8,1,1)
		 */
		UNITY_INTERFACE_EXPORT dim3 getDimBlock() const;


		/**
		 * Get the default dimension grid ((sizeBuffer + 7)/8,1,1)
		 */
		UNITY_INTERFACE_EXPORT dim3 getDimGrid() const;

		/**
		 * Get the size of the buffer
		 */
		UNITY_INTERFACE_EXPORT int getSize() const;




	protected:
		// Pointer to the buffer created in Unity		
		void* _bufferHandle;
		// size of the buffer
		int _size;
		// Resource that can be used to retrieve buffer for CUDA
		cudaGraphicsResource* _graphicsResource;


	private:
		// Default dimension block (8,1,1)
		dim3 _dimBlock;
		// Default dimension grid ((sizeBuffer + 7)/8,1,1)
		dim3 _dimGrid;
};



