#pragma once
#include "texture.h"

#if SUPPORT_D3D11

#include "IUnityGraphicsD3D11.h"
#include "d3d11.h"
#include "renderAPI_D3D11.h"
#include <cassert>
#include <cuda_d3d11_interop.h>

/**
 * @class      Texture_D3D11
 *
 * @brief      This class handles interoperability for texture from Unity to
 *             CUDA, with DX11 graphics API.
 *
 *
 * @see        @ref Texture
 */
class Texture_D3D11 : public Texture
{
    public:
    Texture_D3D11(void *textureHandle, int textureWidth, int textureHeight,
                  int textureDepth, RenderAPI *renderAPI);
    ~Texture_D3D11();

    /**
     * @brief      Register the texture pointer given given in \ref
     * Texture_D3D11 from DirectX to CUDA
     *
     * @return     0 if it works, otherwise something else.
     *
     *
     * @warning    Make sure that the pointer \ref Buffer::_textureHandle is
     * of type \p ID3D11Texture2D, otherwise it can result in a exception.
     */
    int registerTextureInCUDA() override;

    /**
     * @brief      Unregister the texture from DirectX to CUDA
     *
     * @return     0 if it works, otherwise something else.
     */
    int unregisterTextureInCUDA() override;
    /**
     * @brief      Generate mips for texture.
     *
     * @return     0 if it works, otherwise something else.
     */
    int generateMips() override;

    protected:
    /**
     * @brief      Implementation of \ref Texture::copyUnityTextureToAPITexture
     * for DX11.
     *
     * @return     0 if it works, otherwise something else.
     */
    int copyUnityTextureToAPITexture() override;
    /**
     * @brief      Implementation of \ref Texture::copyAPITextureToUnityTexture
     * for DX11.
     *
     * @return     0 if it works, otherwise something else.
     */
    int copyAPITextureToUnityTexture() override;

    private:
    /**
     * @brief      Copy the texture Unity to a buffer.
     *
     * @return     0 if it works, otherwise something else.
     */
    int copyUnityTextureToBuffer();

    ID3D11Texture2D *_texBufferInterop{};
    ID3D11ShaderResourceView *_shaderResources;
    ID3D11Texture2D *_texUnityDX11{};
    RenderAPI_D3D11 *_renderAPI;
};

#endif