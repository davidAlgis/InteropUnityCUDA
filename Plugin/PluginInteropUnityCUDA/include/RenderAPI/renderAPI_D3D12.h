#pragma once
#include "framework.h"
#include "renderAPI.h"

#include <cmath>

// Direct3D 12 implementation of RenderAPI.

#if SUPPORT_D3D12

#include <assert.h>
#include <d3d12.h>
#include "IUnityGraphicsD3D12.h"

class RenderAPI_D3D12 : public RenderAPI
{
    public:
    RenderAPI_D3D12();
    virtual ~RenderAPI_D3D12();

    virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                                    IUnityInterfaces *interfaces);

    private:
    void CreateResources();
    void ReleaseResources();

    private:
    IUnityGraphicsD3D12v2 *s_D3D12;
    ID3D12CommandAllocator *s_D3D12CmdAlloc;
    ID3D12GraphicsCommandList *s_D3D12CmdList;
    UINT64 s_D3D12FenceValue = 0;
    HANDLE s_D3D12Event = NULL;
};

RenderAPI *CreateRenderAPI_D3D12();

const UINT kNodeMask = 0;

#endif // #if SUPPORT_D3D12
