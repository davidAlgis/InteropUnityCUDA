#pragma once
#include "renderAPI.h"
#include "framework.h"
#include "log.h"
// Direct3D 11 implementation of RenderAPI.

#if SUPPORT_D3D11

#include "d3d11.h"
#include "IUnityGraphicsD3D11.h"
#include <assert.h>

class RenderAPI_D3D11 : public RenderAPI
{
    public:
    RenderAPI_D3D11();
    virtual ~RenderAPI_D3D11();
    virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                    IUnityInterfaces *interfaces);
    


    /// <summary>
    /// Create a 2D texture with the given paramaters
    /// return -1, if device has not been set, -2 if dx11
    /// failed to create the texture, else 0
    /// </summary>
    int createTexture2D(int textureWidth, int textureHeight, int textureDepth,
                                     ID3D11Texture2D **textureHandle);

    /// <summary>
    /// Copy the content of texture 2D src to texture 2D dest
    /// This method is UNSAFE, because for efficiency reason it 
    /// doesn't check if src and dest textures are compatible
    /// Therefore, make sure to have compatible textures. See
    /// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-copyresource 
    /// for more details
    /// </summary>
    void copyTextures2D(ID3D11Texture2D* dest, ID3D11Texture2D* src);
    

    private:
    ID3D11Device *_device;
    ID3D11DeviceContext* _context{};
};

RenderAPI *CreateRenderAPI_D3D11();

#endif // #if SUPPORT_D3D11
