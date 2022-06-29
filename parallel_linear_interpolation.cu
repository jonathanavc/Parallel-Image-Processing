// nvcc parallel_linear_interpolation.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0
// nvcc parallel_linear_interpolation.cu -lX11

#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

__device__ int pixel(unsigned char *img, int x, int y, int width, int size, int rgb)
{
    return img[(x) + (y)*width + size * rgb];
}

__global__ void nearest_neighbor_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height)
{
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;

    int scale = (new_width / old_width) + 1;

    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    int r_g_b = blockIdx.z;

    if (pos_x >= new_width || pos_y >= new_height)
        return;

    if (pos_x % scale == 0 && pos_y % scale == 0){ // se mantiene el color ✅
        d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b);
    }
    else if (pos_x % scale == 0){ // se interpola en el eje y ✅
        d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_y % scale) * ((pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
    }
    else if (pos_y % scale == 0){ // se interpola en el eje x ✅
        d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
    }
    else{ // ahora si (creo)✅✅✅✅✅🚬🐛
        // obtener (x,y)
        int x_y_r = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
        // obtener (x,y+1)
        int x_y_1_r = pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale + 1, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b)) / (scale));
        // interpolar
        d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = x_y_r + (pos_y % scale) * ((x_y_1_r - x_y_r) / scale);
    }
}

int main(int argc, char const *argv[])
{
    if (argc != 3)
    {
        cout << "Modo de uso: " << argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: int >= 1)\"" << endl;
        return 1;
    }
    int scale = atoi(argv[2]);
    if (scale < 1)
    {
        cout << "El factor de escalado debe ser un entero mayor o igual a 1" << endl;
        return 1;
    }
    CImg<unsigned char> img_in(argv[1]);
    int old_size = img_in.size() / 3;
    int old_width = img_in.width();
    int old_height = img_in.height();

    unsigned long long size = img_in.size();
    unsigned char *old_image = img_in.data();

    CImg<unsigned char> img_out(old_width * scale - (scale - 1), old_height * scale - (scale - 1), 1, 3, 255);
    unsigned char *new_image = img_out.data();
    int new_size = img_out.size() / 3;
    int new_width = img_out.width();
    int new_height = img_out.height();

    unsigned char *d_old_image;
    unsigned char *d_new_image;

    cudaMalloc((void **)&d_old_image, old_size * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_new_image, new_size * 3 * sizeof(unsigned char));

    cudaMemcpy(d_old_image, old_image, old_size * 3, cudaMemcpyHostToDevice);

    dim3 blkDim(16, 16, 1);
    dim3 grdDim((new_width + 15) / 16, (new_height + 15) / 16, 3);
    nearest_neighbor_interpolation<<<grdDim, blkDim>>>(d_old_image, d_new_image, old_width, old_height, new_width, new_height);

    cudaDeviceSynchronize();

    cudaMemcpy(new_image, d_new_image, new_size * 3, cudaMemcpyDeviceToHost);

    img_out.save("new_img.jpg");

    cudaFree(&d_old_image);
    cudaFree(&new_image);
    return 0;
}
