#include "renderAPI.h"
#include "framework.h"

// Direct3D 11 implementation of RenderAPI.

#if SUPPORT_D3D11

#include <assert.h>
#include <d3d11.h>
#include "IUnityGraphicsD3D11.h"


class RenderAPI_D3D11 : public RenderAPI
{
public:
	RenderAPI_D3D11();
	virtual ~RenderAPI_D3D11() { }

	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces);

private:
	void CreateResources();
	void ReleaseResources();

private:
	ID3D11Device* m_Device;
	ID3D11Buffer* m_VB; // vertex buffer
	ID3D11Buffer* m_CB; // constant buffer
	ID3D11VertexShader* m_VertexShader;
	ID3D11PixelShader* m_PixelShader;
	ID3D11InputLayout* m_InputLayout;
	ID3D11RasterizerState* m_RasterState;
	ID3D11BlendState* m_BlendState;
	ID3D11DepthStencilState* m_DepthState;
};


RenderAPI* CreateRenderAPI_D3D11()
{
	return new RenderAPI_D3D11();
}



RenderAPI_D3D11::RenderAPI_D3D11()
	: m_Device(NULL)
	, m_VB(NULL)
	, m_CB(NULL)
	, m_VertexShader(NULL)
	, m_PixelShader(NULL)
	, m_InputLayout(NULL)
	, m_RasterState(NULL)
	, m_BlendState(NULL)
	, m_DepthState(NULL)
{
}


void RenderAPI_D3D11::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
	switch (type)
	{
	case kUnityGfxDeviceEventInitialize:
	{
		IUnityGraphicsD3D11* d3d = interfaces->Get<IUnityGraphicsD3D11>();
		m_Device = d3d->GetDevice();
		CreateResources();
		break;
	}
	case kUnityGfxDeviceEventShutdown:
		ReleaseResources();
		break;
	}
}


void RenderAPI_D3D11::CreateResources()
{


}


void RenderAPI_D3D11::ReleaseResources()
{
	SAFE_RELEASE(m_VB);
	SAFE_RELEASE(m_CB);
	SAFE_RELEASE(m_VertexShader);
	SAFE_RELEASE(m_PixelShader);
	SAFE_RELEASE(m_InputLayout);
	SAFE_RELEASE(m_RasterState);
	SAFE_RELEASE(m_BlendState);
	SAFE_RELEASE(m_DepthState);
}




#endif // #if SUPPORT_D3D11
