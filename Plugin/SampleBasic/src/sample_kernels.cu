#include "sample_kernels.cuh"

__global__ void writeTex(cudaSurfaceObject_t surf, int width, int height,
                         float time)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {

        float4 t = make_float4(0, abs(cos(time)), 0, 1.0f);

        surf2Dwrite(t, surf, sizeof(float4) * x, y);
    }
}

__global__ void writeTexArray(cudaSurfaceObject_t* surf, int width, int height,
                              int depth, float time)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height)
    {

        float4 t = make_float4(z % 2, abs(cos(time)), 0, 1.0f);

        surf2Dwrite(t, surf[z], sizeof(float4) * x, y);
    }
}

__global__ void writeVertexBuffer(float4 *pos, int size, float time)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // write output vertex
    if (x < size)
    {
        pos[x] = make_float4(cos(2 * CUDART_PI_F * time / x),
                             sin(2 * CUDART_PI_F * time / x), 0.0f, 1.0f);
    }
}

void kernelCallerWriteTexture(const dim3 dimGrid, const dim3 dimBlock,
                              cudaSurfaceObject_t inputSurfaceObj,
                              const float time, const int width,
                              const int height)
{
    writeTex<<<dimGrid, dimBlock>>>(inputSurfaceObj, width, height, time);
}

void kernelCallerWriteTextureArray(const dim3 dimGrid, const dim3 dimBlock,
                                   cudaSurfaceObject_t* inputSurfaceObj,
                                   const float time, const int width,
                                   const int height, const int depth)
{
    writeTexArray<<<dimGrid, dimBlock>>>(inputSurfaceObj, width, height, depth, time);
}

void kernelCallerWriteBuffer(const dim3 dimGrid, const dim3 dimBlock,
                             float4 *vertexPtr, const int size,
                             const float time)
{
    writeVertexBuffer<<<dimGrid, dimBlock>>>(vertexPtr, size, time);
}