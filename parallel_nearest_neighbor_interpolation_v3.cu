// nvcc parallel_nearest_neighbor_interpolation_v3.cu -lX11 -o parallel_NNI
#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"
#include "./others/metrictime.hpp"

using namespace cimg_library;
using namespace std;

static int block_size = 1024;

static int pixel_per_thread = 8;


__global__ void nearest_neighbor_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height, int pixel_per_thread){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;

    float scale = (float)new_width / old_width;
    for (int i = 0; i < pixel_per_thread; i++){
        int pos = pixel_per_thread *(blockIdx.x * blockDim.x + threadIdx.x) + i;
        int pos_x = (pos % new_size) % new_width;
        int pos_y = (pos % new_size) / new_width;

        int r_g_b = pos / new_size;

        if (pos_x >= new_width || pos_y >= new_height)
            continue;
        
        d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = d_old_image[(int)(pos_x / scale) + (int)(pos_y / scale) * old_width + old_size * r_g_b];
    }
}

int main(int argc, char const *argv[]){
    int test = 0;
    if(argc < 3){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: 2, 1.5)\""<<endl;
        return 1;
    }
    
    if(argc > 3){
        if(strcmp(argv[3], "test") == 0){
            cout <<"------------------ Test -------------------" << endl;
            test = 1;
        }
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

    TIMERSTART(parallel_NNI);

    cudaMemcpy(d_old_image, old_image, old_size * 3, cudaMemcpyHostToDevice);

    dim3 blkDim (block_size, 1, 1);
    dim3 grdDim ((((new_size * 3) + block_size - 1)/block_size + pixel_per_thread - 1)/pixel_per_thread, 1, 1);

    nearest_neighbor_interpolation<<<grdDim, blkDim>>>(d_old_image, d_new_image, old_width, old_height, new_width, new_height, pixel_per_thread);

    cudaDeviceSynchronize();

    cudaMemcpy(new_image, d_new_image, new_size * 3, cudaMemcpyDeviceToHost);

    TIMERSTOP(parallel_NNI);

    if(!test) img_out.save("new_img.jpg");

    cudaFree(d_old_image);
    cudaFree(new_image);
    return 0;
}
