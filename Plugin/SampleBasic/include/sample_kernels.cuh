#pragma once
#include "cuda_include.h"
#include "log.h"
#include "math_constants.h"
#include <device_launch_parameters.h>
#include <vector_functions.h>

/**
 * @brief      Write in the second channel of ARGB texture : \f$|\cos(t)\f$
 *
 * @param[in]  dimGrid          The dimension of the grid.
 * You can retrieve a grid dimension associated to your texture in
 * \ref Texture::getDimGrid.
 * @param[in]  dimBlock         The dimension of the block.
 * You can retrieve a block dimension associated to your texture in
 * \ref Texture::getDimBlock.
 * @param[in]  inputSurfaceObj  The input surface object. Use the surface object
 * that has been created with CUDA with \ref Texture::getSurfaceObject.
 * @param[in]  t                The time of execution
 * @param[in]  width            The width of the texture. You can retrieve this
 * data with: \ref Texture::getWidth
 * @param[in]  height           The height of the texture. You can retrieve this
 * data with: \ref Texture::getHeight
 *
 * @see Texture
 *
 * @example action_sample_texture.cpp
 * This file show an example of use of this function.
 */
void kernelCallerWriteTexture(dim3 dimGrid, dim3 dimBlock,
                              cudaSurfaceObject_t inputSurfaceObj, float time,
                              int width, int height);

/**
 * @brief      Call a kernel that write each pixel of each texture of a texture
 * array of ARGB texture with : \f$(z[2], |z+1|\cos(t),0,1)\f$ where \f$z\f$ is
 * the index of the texture in the texture array, and t is the \p time.
 *
 * @param[in]  dimGrid          The dimension of the grid.
 * You can retrieve a grid dimension associated to your texture in
 * \ref Texture::getDimGrid.
 * @param[in]  dimBlock         The dimension of the block.
 * You can retrieve a block dimension associated to your texture in
 * \ref Texture::getDimBlock.
 * @param      surfObjArray  The input surface object. Use the surface object
 * that has been created with CUDA with \ref Texture::getSurfaceObjectArray.
 * @param[in]  time                The time of execution
 * @param[in]  width            The width of the texture. You can retrieve this
 * data with: \ref Texture::getWidth
 * @param[in]  height           The height of the texture. You can retrieve this
 * data with: \ref Texture::getHeight
 * @param[in]  depth         The depth of the texture. You can retrieve this
 * data with \ref Texture::getDepth
 *
 * @see Texture
 *
 * @example action_sample_texture_array.cpp
 * This file show an example of use of this function.
 */
void kernelCallerWriteTextureArray(dim3 dimGrid, dim3 dimBlock,
                                   cudaSurfaceObject_t *surfObjArray,
                                   float time, int width, int height,
                                   int depth);

/**
 * @brief      Write in a vertex buffer the following position :
 * \f[(\cos(\frac{2 \pi t}{|i|+1}), \sin(\frac{2 \pi t}{|i|+1}) 0 ,1)\f]
 *
 * @param[in]  dimGrid          The dimension of the grid.
 * You can retrieve a grid dimension associated to your texture in
 * \ref VertexBuffer::getDimGrid.
 * @param[in]  dimBlock         The dimension of the block.
 * You can retrieve a block dimension associated to your texture in
 * \ref VertexBuffer::getDimBlock.
 * @param      vertexPtr  The vertex pointer.
 * @param[in]  size       The size of the vertex buffer. You can retrieve this
 * data with: \ref VertexBuffer::getSize.
 * @param[in]  time       The time of execution.
 *
 *
 * @see VertexBuffer
 *
 * @example action_sample_vertex_buffer.cpp
 * This file show an example of use of this function.
 */
void kernelCallerWriteBuffer(dim3 dimGrid, dim3 dimBlock, float4 *vertexPtr,
                             int size, float time);