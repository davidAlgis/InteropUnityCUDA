
#include <device_launch_parameters.h>


__global__ void writeTex(cudaSurfaceObject_t surf, int width, int height, float t) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        uchar4 c;
        c.x = 0;
        c.y = (unsigned char)(t*100 + 128) % 255;
        c.z = 0;
        c.w = 255;

        surf2Dwrite(c, surf, 4 * x, y);
    }
}


void kernelCallerWriteTexture(const dim3 dimGrid, const dim3 dimBlock, cudaSurfaceObject_t inputSurfaceObj, const float t, const int width, const int height) 
{
    writeTex << <dimGrid, dimBlock >> > (inputSurfaceObj, width, height, t);

}
