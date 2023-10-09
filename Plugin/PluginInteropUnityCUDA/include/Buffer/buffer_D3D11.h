#pragma once

#include "buffer.h"

#if SUPPORT_D3D11

#include "IUnityGraphicsD3D11.h"
#include <cuda_d3d11_interop.h>
#include <d3d11.h>

/**
 * @class      Buffer_D3D11
 *
 * @brief      This class is a implementation of \ref Buffer for DirectX 11
 * graphics API. Indeed, the register functions depends on the graphics API.
 *
 * @see \ref Buffer
 *
 */
class Buffer_D3D11 : public Buffer
{
    public:
    /**
     * @brief      Constructs a new instance of Buffer for DirectX 11 graphics
     * API
     *
     * @param      bufferHandle  A native pointer of a buffer created in Unity
     * with DirectX 11 graphics API. It must be of type \p ID3D11Buffer.
     *
     * @param[in]  size          The size of the buffer
     *
     * @warning    If the pointer is not of type \p ID3D11Buffer, it can result
     * in a exception.
     */
    Buffer_D3D11(void *bufferHandle, int size);

    /**
     * @brief      Destroys the object.
     */
    ~Buffer_D3D11();

    /**
     * @brief      Register the buffer given in \ref Buffer_D3D11 from DirectX
     * to CUDA
     *
     * @return     0 if it works, otherwise something else.
     *
     *
     * @warning    Make sure that the pointer \ref Buffer::_bufferHandle is
     * of type \p ID3D11Buffer, otherwise it can result in a exception.
     */
    int registerBufferInCUDA() override;

    /**
     * @brief      Unregister the buffer from DirectX to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int unregisterBufferInCUDA() override;
};

#endif
