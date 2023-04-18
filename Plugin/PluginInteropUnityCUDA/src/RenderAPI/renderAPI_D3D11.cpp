#include "RenderAPI_D3D11.h"

#if SUPPORT_D3D11

RenderAPI *CreateRenderAPI_D3D11()
{
    return new RenderAPI_D3D11();
}

RenderAPI_D3D11::RenderAPI_D3D11() : _device(NULL)
{
}

RenderAPI_D3D11::~RenderAPI_D3D11()
{
}

int RenderAPI_D3D11::createTexture2D(int textureWidth, int textureHeight,
                                     int textureDepth,
                                     ID3D11Texture2D **textureHandle)
{
    D3D11_TEXTURE2D_DESC texDesc;
    ZeroMemory(&texDesc, sizeof(texDesc));
    texDesc.Width = textureWidth;
    texDesc.Height = textureHeight;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = textureDepth;
    texDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = 0;
    texDesc.MiscFlags = 0;

    if (_device == NULL)
    {
        Log::log().debugLogError(
            "m_Device has not been initialized in RenderAPI_D3D11, please make "
            "sure event kUnityGfxDeviceEventInitialize has been called "
            "before.");
        return -1;
    }

    HRESULT hr = _device->CreateTexture2D(&texDesc, nullptr, textureHandle);
    if (FAILED(hr))
    {
        Log::log().debugLogError("Error " + std::to_string(hr) +
                                 " when creating Texture in DX11.");
        return -2;
    }
    return 0;
}

int RenderAPI_D3D11::createShaderResource(
    ID3D11Resource *resource, ID3D11ShaderResourceView **shaderResource)
{
    if (_device == NULL)
    {
        Log::log().debugLogError(
            "m_Device has not been initialized in RenderAPI_D3D11, please make "
            "sure event kUnityGfxDeviceEventInitialize has been called "
            "before.");
        return -1;
    }
    ID3D11Texture2D *texUnityDX11 = (ID3D11Texture2D *)resource;
    D3D11_TEXTURE2D_DESC texDesc;
    texUnityDX11->GetDesc(&texDesc);
    // see #6 https://github.com/davidAlgis/InteropUnityCUDA/issues/6
    if (((texDesc.BindFlags & D3D11_BIND_RENDER_TARGET) == false ||
         (texDesc.BindFlags & D3D11_BIND_SHADER_RESOURCE) == false) &&
        (texDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS) == false)
    {
        // see remarks
        // https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-generatemips
        Log::log().debugLogError(
            "The texture wasn't created with the correct bind or misc flags. "
            "Mips cannot be generated. You may have this error, because with "
            "DX11, Texture2D mips cannot be generated");
        return -2;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = -1;

    HRESULT hr =
        _device->CreateShaderResourceView(resource, &srvDesc, shaderResource);
    if (SUCCEEDED(hr) == false)
    {
        Log::log().debugLogError(
            "There has been an error " + std::to_string((int)(hr)) +
            "when creating shader resource view associated to the resource.");
        return -3;
    }

    return 0;
}

void RenderAPI_D3D11::copyTextures2D(ID3D11Texture2D *dest,
                                     ID3D11Texture2D *src)
{
    getCurrentContext()->CopyResource(dest, src);
}

ID3D11DeviceContext *RenderAPI_D3D11::getCurrentContext()
{
    _device->GetImmediateContext(&_context);
    return _context;
}

void RenderAPI_D3D11::ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                         IUnityInterfaces *interfaces)
{
    switch (type)
    {
    case kUnityGfxDeviceEventInitialize: {
        IUnityGraphicsD3D11 *d3d = interfaces->Get<IUnityGraphicsD3D11>();
        _device = d3d->GetDevice();
        break;
    }
    case kUnityGfxDeviceEventShutdown:
        break;
    case kUnityGfxDeviceEventBeforeReset:
        break;
    case kUnityGfxDeviceEventAfterReset:
        break;
    }
}

#endif // #if SUPPORT_D3D11
