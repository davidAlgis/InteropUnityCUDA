#include "framework.h"
#include "renderAPI.h"
#include "log.h"
// Direct3D 11 implementation of RenderAPI.

#if SUPPORT_D3D11

#include <assert.h>
#include <d3d11.h>
#include "IUnityGraphicsD3D11.h"

class RenderAPI_D3D11 : public RenderAPI
{
    public:
    RenderAPI_D3D11();
    virtual ~RenderAPI_D3D11()
    {
    }

    virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                    IUnityInterfaces *interfaces);

    private:
    ID3D11Device *m_Device;
};

RenderAPI *CreateRenderAPI_D3D11()
{
    return new RenderAPI_D3D11();
}

RenderAPI_D3D11::RenderAPI_D3D11() : m_Device(NULL)
{
}

void RenderAPI_D3D11::ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                         IUnityInterfaces *interfaces)
{
    switch (type)
    {
    case kUnityGfxDeviceEventInitialize: {
    	Log::log().debugLog("init dx11");
        IUnityGraphicsD3D11 *d3d = interfaces->Get<IUnityGraphicsD3D11>();
        m_Device = d3d->GetDevice();

        ID3D11DeviceContext *ctx = NULL;
        m_Device->GetImmediateContext(&ctx);
    	Log::log().debugLog("end dx11");
        break;
    }
    case kUnityGfxDeviceEventShutdown:
        break;
    }
}

#endif // #if SUPPORT_D3D11
