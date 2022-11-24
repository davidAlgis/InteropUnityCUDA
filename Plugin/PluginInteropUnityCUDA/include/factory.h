#pragma once

#include "Buffer/vertexBuffer.h"
#include "Texture/texture.h"
#include "framework.h"

namespace Factory {

  VertexBuffer *UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API createBuffer(void *bufferHandle,
                                                                        int size,
                                                                        UnityGfxRenderer apiType);

  Texture *UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API createTexture(void *textureHandle,
                                                                    int textureWidth,
                                                                    int textureHeight,
                                                                    UnityGfxRenderer apiType);
																	
} // namespace Factory
