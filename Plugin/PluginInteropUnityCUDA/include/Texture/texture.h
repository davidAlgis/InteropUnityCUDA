#pragma once
#include "IUnityGraphics.h"
#include "cuda_include.h"
#include "log.h"

class Texture
{
    public:
    /// <summary>
    /// Constructor of texture
    /// </summary>
    /// <param name="textureHandle">A pointer of texture with float4 that has
    /// been generated with Unity (see function GetNativeTexturePtr
    /// https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html)
    /// </param> <param name="textureWidth">the width of the texture</param>
    /// <param name="textureHeight">the height of the texture</param>
    /// <param name="textureDepth">the depth of the texture</param>
    UNITY_INTERFACE_EXPORT Texture(void *textureHandle, int textureWidth,
                                  int textureHeight,
            int textureDepth);

    /// <summary>
    /// Register the texture in CUDA, this has to be override because it depends
    /// on the graphics api
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void registerTextureInCUDA() = 0;

    /// <summary>
    /// Unregister the texture in CUDA, this has to be override because it
    /// depends on the graphics api
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void unRegisterTextureInCUDA() = 0;


    /// <summary>
    /// For some API (DX11) CUDA cannot edit the texture created by Unity
    /// therefore, we have to create a new texture that will used as a buffer
    /// between the unity texture and the surface object that is modify by CUDA
    /// For these API, this function will copy the content of the unity texture
    /// to this buffer, for the other API. It'll do nothing.
    /// If Unity texture has been modify in Unity, you have to do the copy before
    /// reading it in CUDA. It's not necessary if you only write into the texture 
    /// in CUDA, or if the texture has not been modify in Unity.
    /// Tips : not necessary for write only in CUDA or read only in Unity
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void copyUnityTextureToAPITexture() = 0;


    /// <summary>
    /// For some API (DX11) CUDA cannot edit the texture created by Unity
    /// therefore, we have to create a new texture that will used as a buffer
    /// between the unity texture and the surface object that is modify by CUDA
    /// For these API, this function will copy the content of the buffer texture
    /// to the unity texture, for the other API. It'll do nothing.
    /// If API texture has been modify by, you have to do the copy before
    /// reading it in Unity. It's not necessary if you only read into the texture 
    /// in CUDA, or if the texture is only write only in Unity.
    /// Tips : not necessary for read only in CUDA or write only in Unity
    /// </summary>
    UNITY_INTERFACE_EXPORT virtual void copyAPITextureToUnityTexture() = 0;

    /// <summary>
    /// Map a cuda array to the graphics resources and wrap it into a surface
    /// object of cuda
    /// </summary>
    /// <param name="indexInArray"> Array index for array textures or cubemap
    /// face index as defined by cudaGraphicsCubeFace for cubemap textures for
    /// the subresource to access </param> <returns>a cuda surface object on
    /// device memory and which can be edited in cuda</returns>
    UNITY_INTERFACE_EXPORT cudaSurfaceObject_t mapTextureToSurfaceObject(
        int indexInArray = 0);


    /// <summary>
    /// Unmap the cuda array from graphics resources and destroy surface object
    /// This function will wait for all previous GPU activity to complete
    /// </summary>
    /// <param name="inputSurfObj">the surface object that has been created with
    /// <c>mapTextureToSurfaceObject</c> function</param>
    UNITY_INTERFACE_EXPORT void unMapTextureToSurfaceObject(
        cudaSurfaceObject_t &inputSurfObj);

    UNITY_INTERFACE_EXPORT cudaTextureObject_t mapTextureToTextureObject(
        int indexInArray = 0);

    UNITY_INTERFACE_EXPORT void unMapTextureToTextureObject(
        cudaTextureObject_t &texObj);

    /// <summary>
    /// Get the default dimension block (8,8,1)
    /// </summary>
    UNITY_INTERFACE_EXPORT dim3 getDimBlock() const;

    /// <summary>
    /// Get the default dimension grid ((sizeBuffer + 7)/8,((sizeBuffer +
    /// 7)/8,1)
    /// </summary>
    UNITY_INTERFACE_EXPORT dim3 getDimGrid() const;

    /// <summary>
    /// Get the width of the texture
    /// </summary>
    UNITY_INTERFACE_EXPORT int getWidth() const;

    /// <summary>
    /// Get the height of the texture
    /// </summary>
    UNITY_INTERFACE_EXPORT int getHeight() const;

    /// <summary>
    /// Get the depth of the texture
    /// </summary>
    UNITY_INTERFACE_EXPORT int getDepth() const;

    /// <summary>
    /// Get the native texture pointer
    /// </summary>
    UNITY_INTERFACE_EXPORT void *getNativeTexturePtr() const;

    protected:
    // Pointer to the texture created in Unity
    void *_textureHandle;
    // width of the texture
    int _textureWidth;
    // height of the texture
    int _textureHeight;
    // depth of the texture <2 <=> to Texture2D; >1 <=> Texture2DArray
    int _textureDepth;
    // Resource that can be used to retrieve the surface object for CUDA
    cudaGraphicsResource *_pGraphicsResource;

    private:
    dim3 _dimBlock;
    dim3 _dimGrid;
};
