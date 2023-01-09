#pragma once
#include "log.h"
#include "cudaInclude.h"
#include "texture.h"

UNITY_INTERFACE_EXPORT struct dim3;

class UNITY_INTERFACE_EXPORT VertexBuffer
{
	public:

		/// <summary>
		/// Constructor of vertex buffer
		/// </summary>
		/// <param name="bufferHandle">A pointer of computeBuffer with float4 that has been generated with Unity (see function
		/// GetNativeBufferPtr https://docs.unity3d.com/ScriptReference/ComputeBuffer.GetNativeBufferPtr.html) </param>
		/// <param name="size">the size of the computeBuffer</param>
		VertexBuffer(void* bufferHandle, int size);

		/// <summary>
		/// Register the buffer in CUDA, this has to be override because it depends on the graphics api
		/// </summary>
		virtual void registerBufferInCUDA() = 0;

		/// <summary>
		/// Unregisteregister the buffer in CUDA, this has to be override because it depends on the graphics api
		/// </summary>
		virtual void unRegisterBufferInCUDA() = 0;


		/// <summary>
		/// Map resources to CUDA
		/// </summary>
		/// <returns>an array of float4* defined on device memory and which can be edited in cuda</returns>
		template <typename T>
		T* mapResources()
		{
			// map resource
			CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource, 0));
			// pointer toward an array of float4 on device memory : the compute buffer
			T* vertexPtr;
			// number of bytes that has been readed
			size_t numBytes;
			// map the resources on a float4 array that can be modify on device
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertexPtr, &numBytes,
				_pGraphicsResource));
			return vertexPtr;
		}

		/// <summary>
		/// Unmap resources from CUDA
		/// This function will wait for all previous GPU activity to complete
		/// </summary>
		void unmapResources();

		/// <summary>
		/// Get the default dimension block (8,1,1)
		/// </summary>
		dim3 getDimBlock() const;


		/// <summary>
		/// Get the default dimension grid ((sizeBuffer + 7)/8,1,1)
		/// </summary>
		dim3 getDimGrid() const;

		/// <summary>
		/// Get the size of the buffer
		/// </summary>
		int getSize() const;




	protected:
		// Pointer to the buffer created in Unity		
		void* _bufferHandle;
		// size of the buffer
		int _size;
		// Resource that can be used to retrieve buffer for CUDA
		cudaGraphicsResource* _pGraphicsResource;


	private:
		// Default dimension block (8,1,1)
		dim3 _dimBlock;
		// Default dimension grid ((sizeBuffer + 7)/8,1,1)
		dim3 _dimGrid;
};



