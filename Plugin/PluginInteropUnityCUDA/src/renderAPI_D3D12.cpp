#include "renderAPI.h"
#include "framework.h"

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
	virtual ~RenderAPI_D3D12() { }

	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces);

private:
	UINT64 AlignPow2(UINT64 value);
	UINT64 GetAlignedSize(int width, int height, int pixelSize, int rowPitch);
	ID3D12Resource* GetUploadResource(UINT64 size);
	void CreateResources();
	void ReleaseResources();

private:
	IUnityGraphicsD3D12v2* s_D3D12;
	ID3D12Resource* s_D3D12Upload;
	ID3D12CommandAllocator* s_D3D12CmdAlloc;
	ID3D12GraphicsCommandList* s_D3D12CmdList;
	UINT64 s_D3D12FenceValue = 0;
	HANDLE s_D3D12Event = NULL;
};


RenderAPI* CreateRenderAPI_D3D12()
{
	return new RenderAPI_D3D12();
}


const UINT kNodeMask = 0;


RenderAPI_D3D12::RenderAPI_D3D12()
	: s_D3D12(NULL)
	, s_D3D12Upload(NULL)
	, s_D3D12CmdAlloc(NULL)
	, s_D3D12CmdList(NULL)
	, s_D3D12FenceValue(0)
	, s_D3D12Event(NULL)
{
}

UINT64 RenderAPI_D3D12::AlignPow2(UINT64 value)
{
	UINT64 aligned = pow(2, (int)log2(value));
	return aligned >= value ? aligned : aligned * 2;
}

UINT64 RenderAPI_D3D12::GetAlignedSize( int width, int height, int pixelSize, int rowPitch)
{
	UINT64 size = width * height * pixelSize;

	size = AlignPow2(size);

	if (size < D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT)
	{
		return D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
	}
	else if (width * pixelSize < rowPitch)
	{
		return rowPitch * height;
	}
	else
	{
		return size;
	}
}

ID3D12Resource* RenderAPI_D3D12::GetUploadResource(UINT64 size)
{
	if (s_D3D12Upload)
	{
		D3D12_RESOURCE_DESC desc = s_D3D12Upload->GetDesc();
		if (desc.Width == size)
			return s_D3D12Upload;
		else
			s_D3D12Upload->Release();
	}

	// Texture upload buffer
	D3D12_HEAP_PROPERTIES heapProps = {};
	heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
	heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
	heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
	heapProps.CreationNodeMask = kNodeMask;
	heapProps.VisibleNodeMask = kNodeMask;

	D3D12_RESOURCE_DESC heapDesc = {};
	heapDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	heapDesc.Alignment = 0;
	heapDesc.Width = size;
	heapDesc.Height = 1;
	heapDesc.DepthOrArraySize = 1;
	heapDesc.MipLevels = 1;
	heapDesc.Format = DXGI_FORMAT_UNKNOWN;
	heapDesc.SampleDesc.Count = 1;
	heapDesc.SampleDesc.Quality = 0;
	heapDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	heapDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

	ID3D12Device* device = s_D3D12->GetDevice();
	HRESULT hr = device->CreateCommittedResource(
		&heapProps,
		D3D12_HEAP_FLAG_NONE,
		&heapDesc,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&s_D3D12Upload));
	if (FAILED(hr))
	{
		OutputDebugStringA("Failed to CreateCommittedResource.\n");
	}

	return s_D3D12Upload;
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
	SAFE_RELEASE(s_D3D12Upload);
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
	}
}

#endif // #if SUPPORT_D3D12
