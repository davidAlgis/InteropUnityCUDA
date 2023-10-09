#pragma once
#include "framework.h"
#include "renderAPI.h"

#include <cmath>

// Direct3D 12 implementation of RenderAPI.

#if SUPPORT_D3D12

#include <cassert>
#include <d3d12.h>

// make sure you include d3d12.h before including IUnityGraphicsD3D12.h
#include "IUnityGraphicsD3D12.h"

class RenderAPI_D3D12 : public RenderAPI
{
    public:
    RenderAPI_D3D12();
    ~RenderAPI_D3D12() override;

    void ProcessDeviceEvent(UnityGfxDeviceEventType type,
                            IUnityInterfaces *interfaces) override;

    private:
    void CreateResources();
    void ReleaseResources();

    IUnityGraphicsD3D12v2 *s_D3D12;
    ID3D12CommandAllocator *s_D3D12CmdAlloc;
    ID3D12GraphicsCommandList *s_D3D12CmdList;
    UINT64 s_D3D12FenceValue = 0;
    HANDLE s_D3D12Event = nullptr;
};

RenderAPI *CreateRenderAPI_D3D12();

const UINT kNodeMask = 0;

#endif // #if SUPPORT_D3D12
