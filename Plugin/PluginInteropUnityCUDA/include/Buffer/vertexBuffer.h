#pragma once
#include "log.h"
#include "cudaInclude.h"

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
		float4* mapResources();

		/// <summary>
		/// Unmap resources from CUDA
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



