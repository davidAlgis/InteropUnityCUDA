#pragma once
#include "IUnityGraphics.h"
#include "cuda_include.h"
#include "log.h"
#include <string.h>

/**
 * @class      Texture
 *
 * @brief      This class give you a texture for Unity/CUDA interoperability.
 * It's a base class for each graphics API that is implemented.
 *
 * @see \ref Texture_D3D11, \ref Texture_OpenGLCoreES
 *
 * @example action_sample_texture.cpp shows an example of use of this
 * class with a single texture.
 *
 *
 * @example action_sample_texture_array.cpp shows an example of use of this
 * class with a texture array.
 */
class Texture
{
    public:
    /**
     * @brief      Constructor of texture
     *
     * @param      textureHandle  A pointer of texture with float4 that has been
     *                            generated with Unity (see function
     *                            GetNativeTexturePtr
     *                            https://docs.unity3d.com/ScriptReference/Texture.GetNativeTexturePtr.html)
     * @param      textureWidth   the width of the texture
     * @param      textureHeight  the height of the texture
     * @param      textureDepth   the depth of the texture (should be greater
     *                            than 0)
     *
     * @example action_sample_texture.cpp shows an example of use of this
     * class with a single texture (\p textureDepth = 1)
     *
     *
     * @example action_sample_texture_array.cpp shows an example of use of this
     * class with a texture array. (\p textureDepth > 1)
     */
    UNITY_INTERFACE_EXPORT Texture(void *textureHandle, int textureWidth,
                                   int textureHeight, int textureDepth);

    /**
     * @brief      Destructor of texture, it will free \ref _surfObjArray
     */
    UNITY_INTERFACE_EXPORT ~Texture();

    /**
     * @brief      Register the texture in CUDA, this has to be override because
     *             it depends on the graphics api
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT virtual int registerTextureInCUDA() = 0;

    /**
     * @brief      Unregistered the texture in CUDA, this has to be override
     *             because it depends on the graphics api
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT virtual int unregisterTextureInCUDA() = 0;

    /**
     * @brief      For some API (DX11) CUDA cannot edit the texture created by
     *             Unity therefore, we have to create a new texture that will
     *             used as a buffer between the unity texture and the surface
     *             object that is modify by CUDA For these API, this function
     *             will copy the content of the unity texture to this buffer,
     *             for the other API. It'll do nothing. If Unity texture has
     *             been modify in Unity, you have to do the copy before reading
     *             it in CUDA. It's not necessary if you only write into the
     *             texture  in CUDA, or if the texture has not been modify in
     *             Unity.
     *
     * @note       Not necessary for write only in CUDA or read
     *             only in Unity.
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT virtual int copyUnityTextureToAPITexture() = 0;

    /**
     * @brief      For some API (DX11) CUDA cannot edit the texture created by
     *             Unity therefore, we have to create a new texture that will
     *             used as a buffer between the unity texture and the surface
     *             object that is modify by CUDA For these API, this function
     *             will copy the content of the buffer texture to the unity
     *             texture, for the other API. It'll do nothing. If API texture
     *             has been modify by, you have to do the copy before reading it
     *             in Unity. It's not necessary if you only read into the
     *             texture in CUDA, or if the texture is only write only in
     *             Unity.
     *
     * @note       Not necessary for read only in CUDA or write
     *             only in Unity.
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT virtual int copyAPITextureToUnityTexture() = 0;

    /**
     * @brief      Map the cuda array from the graphics resources and create a
     *             surface object from it. To write into the surface object use
     *             the getter of _surfObjArray.
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT int mapTextureToSurfaceObject();

    /**
     * @brief      Unmap the cuda array from graphics resources and destroy
     *             surface object This function will wait for all previous GPU
     *             activity to complete
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT int unmapTextureToSurfaceObject();

    /**
     * @brief      Generate the mips maps of the texture For DX11 it doesn't
     *             works for Texture2D see issue #6
     *             https://github.com/davidAlgis/InteropUnityCUDA/issues/6
     *
     * @return     0 if it works, otherwise something else.
     */
    UNITY_INTERFACE_EXPORT virtual int generateMips() = 0;

    /**
     * @brief      Get the default dimension block \f$(8,8,1)\f$
     *
     * @return      The dimension of the block.
     */
    UNITY_INTERFACE_EXPORT dim3 getDimBlock() const;

    /**
     * @brief      Get the default dimension grid
     * \f$((sizeBuffer + 7)/8,((sizeBuffer + 7)/8,1)\f$
     *
     * @return     The dimension of the grid.
     */
    UNITY_INTERFACE_EXPORT dim3 getDimGrid() const;

    /**
     * @brief      Get the width of the texture
     *
     * @return     The width of the texture
     */
    UNITY_INTERFACE_EXPORT int getWidth() const;

    /**
     * @brief      Get the height of the texture
     *
     * @return     The height of the texture.
     */
    UNITY_INTERFACE_EXPORT int getHeight() const;

    /**
     * @brief      Get the depth of the texture
     *
     * @return     The depth of the texture.
     */
    UNITY_INTERFACE_EXPORT int getDepth() const;

    /**
     * @brief      Get the native texture pointer
     *
     * @return     The native texture pointer.
     */
    UNITY_INTERFACE_EXPORT void *getNativeTexturePtr() const;

    /**
     * @brief      Get the pointer of d_surfObjArray This array of surface
     *             object is necessary to write or read into a texture
     * @return     The surface object array that will be used to write into the
     * texture with CUDA.
     *
     * @example action_sample_texture.cpp shows an example of use of this
     * class with a single texture.
     */
    UNITY_INTERFACE_EXPORT cudaSurfaceObject_t *getSurfaceObjectArray() const;

    /**
     * @brief      Get the surface object associated to the given indexInArray
     *             This array of surface object is necessary to write or read
     *             into a texture
     * @param      indexInArray  index of the texture to get the surface object
     * (must be between 0 and textureDepth)
     * @return     the surface object associated to it. */
    UNITY_INTERFACE_EXPORT cudaSurfaceObject_t
    getSurfaceObject(int indexInArray = 0) const;

    protected:
    /**
     * @brief     Pointer to the texture created in Unity
     */
    void *_textureHandle;

    /**
     * @brief     Width of the texture
     */
    int _textureWidth;

    /**
     * @brief     Height of the texture
     */
    int _textureHeight;

    /**
     * @brief     Depth of the texture <2 <=> to Texture2D; >1 <=>
     * Texture2DArray
     */
    int _textureDepth;

    /**
     * @brief     Resource that can be used to retrieve the surface object for
     * CUDA
     */
    cudaGraphicsResource *_graphicsResource;

    private:
    /**
     * @brief      An array of surface object that will be of the size of
     * texture depth This array is allocate on host side and will be copy to
     * device memory when texture is map to it
     */
    cudaSurfaceObject_t *_surfObjArray;

    /**
     * @brief     A device array of surface object that will be of the size of
     * texture depth This array is allocate on device memory. the surface object
     * is the object that you can used to write into texture from cuda api (eg.
     * with surf2DWrite)
     */
    cudaSurfaceObject_t *d_surfObjArray;

    /**
     * @brief     Default dimension block \f$(8,1,1)\f$
     */
    dim3 _dimBlock;

    /**
     * @brief     Default dimension grid \f$((sizeBuffer + 7)/8,1,1)\f$
     */
    dim3 _dimGrid;
};
