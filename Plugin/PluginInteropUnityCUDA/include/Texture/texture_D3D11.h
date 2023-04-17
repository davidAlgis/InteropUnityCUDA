#pragma once
#include "texture.h"

#if SUPPORT_D3D11

#include "IUnityGraphicsD3D11.h"
#include "d3d11.h"
#include "renderAPI_D3D11.h"
#include <assert.h>
#include <cuda_d3d11_interop.h>

/**
 * @brief      This class handles interoperability for texture from Unity to
 * CUDA, with dx11 graphics API. With dx11 interoperability work differently,
 * texture created in Unity, cannot be directly registered in CUDA, therefore we
 * need to create another texture
 * (_texBufferInterop) that will be used as a buffer. We will copy the content
 * of the texture created in Unity and then we will registered this new texture
 * in CUDA see issue #2 on github for more details
 */
class Texture_D3D11 : public Texture
{
    public:
    Texture_D3D11(void *textureHandle, int textureWidth, int textureHeight,
                  int textureDepth, RenderAPI *renderAPI);
    ~Texture_D3D11();

    virtual int registerTextureInCUDA();
    virtual int unregisterTextureInCUDA();
    virtual int generateMips();

    protected:
    virtual int copyUnityTextureToAPITexture();
    virtual int copyAPITextureToUnityTexture();

    private:
    int copyUnityTextureToBuffer();

    ID3D11Texture2D *_texBufferInterop{};
    ID3D11ShaderResourceView *_shaderResources;
    ID3D11Texture2D *_texUnityDX11{};
    RenderAPI_D3D11 *_renderAPI;
};

#endif