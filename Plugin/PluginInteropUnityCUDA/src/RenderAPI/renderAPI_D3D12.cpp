#include "renderAPI_D3D12.h"

#if SUPPORT_D3D12

RenderAPI* CreateRenderAPI_D3D12()
{
	return new RenderAPI_D3D12();
}



RenderAPI_D3D12::RenderAPI_D3D12()
	: s_D3D12(NULL)
	, s_D3D12CmdAlloc(NULL)
	, s_D3D12CmdList(NULL)
	, s_D3D12FenceValue(0)
	, s_D3D12Event(NULL)
{
}

RenderAPI_D3D12::~RenderAPI_D3D12()
{
	
}

void RenderAPI_D3D12::CreateResources()
{
	ID3D12Device* device = s_D3D12->GetDevice();

	HRESULT hr = E_FAIL;

	// Command list
	hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_D3D12CmdAlloc));
	if (FAILED(hr)) OutputDebugStringA("Failed to CreateCommandAllocator.\n");
	hr = device->CreateCommandList(kNodeMask, D3D12_COMMAND_LIST_TYPE_DIRECT, s_D3D12CmdAlloc, nullptr, IID_PPV_ARGS(&s_D3D12CmdList));
	if (FAILED(hr)) OutputDebugStringA("Failed to CreateCommandList.\n");
	s_D3D12CmdList->Close();

	// Fence
	s_D3D12FenceValue = 0;
	s_D3D12Event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}


void RenderAPI_D3D12::ReleaseResources()
{
	if (s_D3D12Event)
		CloseHandle(s_D3D12Event);
	SAFE_RELEASE(s_D3D12CmdList);
	SAFE_RELEASE(s_D3D12CmdAlloc);
}


void RenderAPI_D3D12::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
	switch (type)
	{
	case kUnityGfxDeviceEventInitialize:
		s_D3D12 = interfaces->Get<IUnityGraphicsD3D12v2>();
		CreateResources();
		break;
	case kUnityGfxDeviceEventShutdown:
		ReleaseResources();
		break;
    case kUnityGfxDeviceEventBeforeReset:
    case kUnityGfxDeviceEventAfterReset:
        break;
    }
}

#endif // #if SUPPORT_D3D12
