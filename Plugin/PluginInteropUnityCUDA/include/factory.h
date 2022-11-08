#pragma once
class Buffer;
class Texture;

namespace Factory
{
	Buffer* createBuffer(void* bufferHandle, int size, int stride, UnityGfxRenderer apiType);
	Texture* createTexture(void* textureHandle, int textureWidth, int textureHeight, UnityGfxRenderer apiType);
}

