#include "vertexBuffer_OpenGLCoreES.h"
#include "texture_OpenGLCoreES.h"

namespace Factory
{
	VertexBuffer* createBuffer(void* bufferHandle, int size, UnityGfxRenderer apiType)
	{
		VertexBuffer* buffer = NULL;
		// if SUPPORT_OPENGL_UNIFIED
#if SUPPORT_OPENGL_UNIFIED
		if (apiType == kUnityGfxRendererOpenGLCore || apiType == kUnityGfxRendererOpenGLES20 || apiType == kUnityGfxRendererOpenGLES30)
		{
			buffer = new VertexBuffer_OpenGLCoreES(bufferHandle, size);
		}
#endif 

		// Unknown or unsupported graphics API
		return buffer;
	}


	Texture* createTexture(void* textureHandle, int textureWidth, int textureHeight, int textureDepth, UnityGfxRenderer apiType)
	{
		Texture* texture = NULL;
		//#	if SUPPORT_D3D11
		//	if (apiType == kUnityGfxRendererD3D11)
		//	{
		//		extern Texture* CreateRenderAPI_D3D11();
		//		return CreateRenderAPI_D3D11();
		//	}
		//#	endif // if SUPPORT_D3D11
		//
		//#	if SUPPORT_D3D12
		//	if (apiType == kUnityGfxRendererD3D12)
		//	{
		//		extern RenderAPI* CreateRenderAPI_D3D12();
		//		return CreateRenderAPI_D3D12();
		//	}
		//#	endif // if SUPPORT_D3D12

		// if SUPPORT_OPENGL_UNIFIED
#if SUPPORT_OPENGL_UNIFIED
		if (apiType == kUnityGfxRendererOpenGLCore || apiType == kUnityGfxRendererOpenGLES20 || apiType == kUnityGfxRendererOpenGLES30)
		{
			texture = new Texture_OpenGLCoreES(textureHandle, textureWidth, textureHeight, textureDepth);
		}
#endif 

		// will be NULL is unknown or unsupported graphics API
		return texture;
	}


}