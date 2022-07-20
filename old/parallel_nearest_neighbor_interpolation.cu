// nvcc parallel_nearest_neighbor_interpolation.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0
// nvcc parallel_nearest_neighbor_interpolation.cu -lX11
#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

__global__ void nearest_neighbor_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;

    float scale = (float)new_width / old_width;
    
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= new_width || pos_y >= new_height)
        return;

    // R
    d_new_image[pos_x + pos_y * new_width] = d_old_image[(int)(pos_x / scale) + (int)(pos_y / scale) * old_width];
    // G
    d_new_image[pos_x + pos_y * new_width + new_size] = d_old_image[(int)(pos_x / scale) + (int)(pos_y / scale) * old_width + old_size];
    // B
    d_new_image[pos_x + pos_y * new_width + new_size * 2] = d_old_image[(int)(pos_x / scale) + (int)(pos_y / scale) * old_width + old_size * 2];
}

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        cout << "Modo de uso: " << argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: 2, 1.5)\"" << endl;
        return 1;
    }
    float scale = atof(argv[2]);
    CImg<unsigned char> img_in(argv[1]);
    int old_size = img_in.size() / 3;
    int old_width = img_in.width();
    int old_height = img_in.height();

    unsigned long long size = img_in.size();
    unsigned char *old_image = img_in.data();

    CImg<unsigned char> img_out(old_width * scale, old_height * scale, 1, 3, 255);
    unsigned char *new_image = img_out.data();
    int new_size = img_out.size() / 3;
    int new_width = img_out.width();
    int new_height = img_out.height();

    unsigned char *d_old_image;
    unsigned char *d_new_image;

    cudaMalloc((void**)&d_old_image, old_size * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_new_image, new_size * 3 * sizeof(unsigned char));

    cudaMemcpy(d_old_image, old_image, old_size * 3, cudaMemcpyHostToDevice);

    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((new_width + 15)/16, (new_height+ 15)/16, 1);
    nearest_neighbor_interpolation<<<grdDim, blkDim>>>(d_old_image, d_new_image, old_width, old_height, new_width, new_height);

    cudaDeviceSynchronize();

    cudaMemcpy(new_image, d_new_image, new_size * 3, cudaMemcpyDeviceToHost);

    img_out.save("new_img.jpg");

    cudaFree(&d_old_image);
    cudaFree(&new_image);
    return 0;
}
