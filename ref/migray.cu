//nvcc -o migray migray.cu -lX11
#include <iostream>
#include <cuda_runtime.h>
#include "CImg.h"

using namespace std;
using namespace cimg_library;

__global__ void rgb2gray(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;

    /*
     * CImg canales RGB estan divididos no entrelazados.
     * esto es RRRR.... GGGG.... BBBB... y no RBGRGBRGB....
     * (http://cimg.eu/reference/group__cimg__storage.html)
     */
    unsigned char r = d_src[pos_y * width + pos_x];
    unsigned char g = d_src[(height + pos_y ) * width + pos_x];
    unsigned char b = d_src[(height * 2 + pos_y) * width + pos_x];

    unsigned int _gray = (unsigned int)((float)(r + g + b) / 3.0f + 0.5);
    unsigned char gray = _gray > 255 ? 255 : _gray;

    d_dst[pos_y * width + pos_x] = gray;
}


int main(int argc, char *argv[])
{
    if(argc < 2){
	cout<<" uso: "<<argv[0]<<" img"<<endl;
	return 1;
    }
    //carga imagen
    CImg<unsigned char> src(argv[1]);
    int width = src.width();
    int height = src.height();
    unsigned long size = src.size();

    //crea puntero a imagen 
    unsigned char *h_src = src.data();

    CImg<unsigned char> dst(width, height, 1, 1);
    unsigned char *h_dst = dst.data();

    unsigned char *d_src;
    unsigned char *d_dst;

    cudaMalloc((void**)&d_src, size);
    cudaMalloc((void**)&d_dst, width*height*sizeof(unsigned char));

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    //lanza el kernel
    dim3 blkDim (16, 16, 1);
    dim3 grdDim ((width + 15)/16, (height + 15)/16, 1);
    rgb2gray<<<grdDim, blkDim>>>(d_src, d_dst, width, height);

    // espera a que termine kernel
    cudaDeviceSynchronize();

    // copia resultado de GPU a Host
    cudaMemcpy(h_dst, d_dst, width*height, cudaMemcpyDeviceToHost);

    char imgout[30] = "salida.jpg";
    dst.save(imgout);

    cudaFree(d_src);
    cudaFree(d_dst);
    
    CImg<unsigned char> src_gray(imgout);
    CImgDisplay main_disp(src_gray, "despues de procesar");
    while (!main_disp.is_closed())
        main_disp.wait();

    return 0;
}
