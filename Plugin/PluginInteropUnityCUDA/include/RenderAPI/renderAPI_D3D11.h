#pragma once
#include "framework.h"
#include "log.h"
#include "renderAPI.h"

#if SUPPORT_D3D11
#include "IUnityGraphicsD3D11.h"
#include "d3d11.h"
#include <cassert>

/**
 * @class      RenderAPI_D3D11
 *
 * @brief      This class describes the implementation of \ref RenderAPI for
 * DirectX 11 graphics API.
 */
class RenderAPI_D3D11 : public RenderAPI
{
    public:
    /**
     * @brief      Constructs a new instance.
     */
    RenderAPI_D3D11();
    ~RenderAPI_D3D11() override;
    void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                            IUnityInterfaces *interfaces) override;

    /**
     * @brief      Creates a shader resource with current context
     *
     * @param      resource        The resource which is associated to shader
     *                             resource
     * @param      shaderResource  The shader resource to create (out param)
     *
     * @return     -1, if device has not been set, -2 if DirectX 11 failed to
     * create the texture, else 0
     */
    int createShaderResource(ID3D11Resource *resource,
                             ID3D11ShaderResourceView **shaderResource);

    /**
     * @brief      Create a 2D texture with the given parameters
     *
     * @param[in]  textureWidth   The texture width
     * @param[in]  textureHeight  The texture height
     * @param[in]  textureDepth   The texture depth
     * @param      textureHandle  The texture handle
     *
     * @return     -1, if device has not been set, -2 if DirectX 11 failed to
     * create the texture, else 0
     */
    int createTexture2D(int textureWidth, int textureHeight, int textureDepth,
                        ID3D11Texture2D **textureHandle);

    /**
     * @brief       Copy the content of texture 2D src to texture 2D dest
     * This method is UNSAFE, because for efficiency reason it
     * doesn't check if src and dest textures are compatible
     * Therefore, make sure to have compatible textures. See
     * https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-copyresource
     * for more details.
     *
     * @param      dest  The destination texture
     * @param      src   The source texture
     */
    void copyTextures2D(ID3D11Texture2D *dest, ID3D11Texture2D *src);

    /**
     * @brief      Gets the current context.
     *
     * @return     The current context.
     */
    ID3D11DeviceContext *getCurrentContext();

    private:
    ID3D11Device *_device;
    ID3D11DeviceContext *_context{};
};

/**
 * @brief      Create a graphics API implementation instance for the DirectX 11.
 *
 * @return     A pointer toward the RenderAPI object.
 */
RenderAPI *CreateRenderAPI_D3D11();

#endif // #if SUPPORT_D3D11
