#pragma once
// Direct3D 11 implementation of Vertex Buffer API
#include "buffer.h"

#if SUPPORT_D3D11

#include "IUnityGraphicsD3D11.h"
#include <cuda_d3d11_interop.h>
#include <d3d11.h>

class VertexBuffer_D3D11 : public Buffer
{
    public:
    VertexBuffer_D3D11(void *bufferHandle, int size);
    ~VertexBuffer_D3D11();

    /**
     * @brief      Register the buffer from OpenGL to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int registerBufferInCUDA() override;

    /**
     * @brief      Unregister the buffer from OpenGL to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int unregisterBufferInCUDA() override;
};

#endif
