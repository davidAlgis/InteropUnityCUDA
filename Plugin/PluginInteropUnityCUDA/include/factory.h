#pragma once
class VertexBuffer;
class Texture;

namespace Factory
{
	VertexBuffer* createBuffer(void* bufferHandle, int size, UnityGfxRenderer apiType);
	Texture* createTexture(void* textureHandle, int textureWidth, int textureHeight, UnityGfxRenderer apiType);
}

