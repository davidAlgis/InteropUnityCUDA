#pragma once
#include "texture_OpenGLCoreES.h"
#include "vertex_buffer_OpenGLCoreES.h"
#include "texture_D3D11.h"
#include "vertex_buffer_D3D11.h"
#include "renderAPI.h"


/// <summary>
/// Simple factory template to create some vertex buffer or texture in function of API type 
/// </summary>
namespace Factory
{
	VertexBuffer* createBuffer(void* bufferHandle, int size, UnityGfxRenderer apiType, RenderAPI* renderAPI);
	Texture* createTexture(void* textureHandle, int textureWidth, int textureHeight, int textureDepth, UnityGfxRenderer apiType, RenderAPI* renderAPI);
}

