#pragma once
// OpenGL Core profile (desktop) or OpenGL ES (mobile) implementation of
// Textures API. Supports several flavors: Core, ES2, ES3
#include "texture.h"

#if SUPPORT_OPENGL_UNIFIED

#include "openGL_include.h"
#include <cuda_gl_interop.h>

template <class T> class Texture_OpenGLCoreES : public Texture<T>
{
    public:
    Texture_OpenGLCoreES(void *textureHandle, int textureWidth,
                         int textureHeight, int textureDepth)
        : Texture(textureHandle, textureWidth, textureHeight, textureDepth)
    {
    }

    ~Texture_OpenGLCoreES()
    {
        GL_CHECK();
        CUDA_CHECK(cudaGetLastError());
    };

    /// <summary>
    /// Has to be call after the first issue plugin event
    /// see. https://docs.unity3d.com/ScriptReference/GL.IssuePluginEvent.html
    /// register a graphics resources defined from the texture openGL
    /// </summary>
    virtual void registerTextureInCUDA()
    {
        // if depth is < 2 it's a texture2D, else it's a texture2DArray
        GLenum target = _textureDepth < 2 ? GL_TEXTURE_2D : GL_TEXTURE_3D;

        // cast the pointer on the texture of unity to gluint
        GLuint gltex = (GLuint)(size_t)(_textureHandle);
        // glBindTexture(target, gltex);
        GL_CHECK();
        // register the texture to cuda : it initialize the _pGraphicsResource
        CUDA_CHECK(
            cudaGraphicsGLRegisterImage(&_pGraphicsResource, gltex, target,
                                        cudaGraphicsRegisterFlagsWriteDiscard));
        kernelCallerCreateSurfaceWrapper_OpenGLCoreES(_surfaceWrapper,
                                              _pGraphicsResource);
    }

    virtual void unRegisterTextureInCUDA()
    {
        kernelCallerDeleteSurfaceWrapper(_surfaceWrapper);
        CUDA_CHECK(cudaGraphicsUnregisterResource(_pGraphicsResource));
    }

    virtual void mapTextureToSurfaceObject()
    {
        // for (int indexInArray = 0; indexInArray < _textureDepth; indexInArray++)
        // {
        //      // map the resource to cuda
        //     CUDA_CHECK(cudaGraphicsMapResources(1, &_pGraphicsResource));
        //     // cuda array on which the resources will be sended
        //     cudaArray *arrayPtr;
        //     // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031
        //     CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(
        //         &arrayPtr, _pGraphicsResource, indexInArray, 0));

        //     // Wrap the cudaArray in a surface object
        //     cudaResourceDesc resDesc;
        //     memset(&resDesc, 0, sizeof(resDesc));
        //     resDesc.resType = cudaResourceTypeArray;
        //     resDesc.res.array.array = arrayPtr;
        //     CUDA_CHECK(cudaCreateSurfaceObject(&_surfObjectsArray[indexInArray], &resDesc));
        //     CUDA_CHECK(cudaGetLastError());
        // } 
    }

    protected:
    virtual void copyUnityTextureToAPITexture()
    {
    }

    virtual void copyAPITextureToUnityTexture()
    {
    }
};

#endif