#pragma once
class VertexBuffer;
class Texture;

/// <summary>
/// Simple factory template to create some vertex buffer or texture in function of API type 
/// </summary>
namespace Factory
{
	VertexBuffer* createBuffer(void* bufferHandle, int size, UnityGfxRenderer apiType);
	Texture* createTexture(void* textureHandle, int textureWidth, int textureHeight, UnityGfxRenderer apiType);
}

