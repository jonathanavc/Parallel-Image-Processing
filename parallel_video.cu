// nvcc parallel_video.cu -lX11


#include <iostream>
#include <vector>
#include <dirent.h>
#include <cuda_runtime.h>
#include "CImg.h"
#include "./others/metrictime.hpp"

using namespace cimg_library;
using namespace std;

static int block_size = 1024;

static int pixel_per_thread = 32;

static int img_per_kernel = 2;

__device__ int pixel(unsigned char *img, int x, int y, int width, int size, int rgb){
    return img[(x) + (y)*width + size * rgb];
}

__global__ void linear_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height,int  scale,int pixel_per_thread){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;
    for (int i = 0; i < pixel_per_thread; i++){
        int pos = pixel_per_thread *(blockIdx.x * blockDim.x + threadIdx.x) + i;
        int pos_x = (pos % new_size) % new_width;
        int pos_y = (pos % new_size) / new_width;
        int r_g_b = pos / new_size;
        if (pos_x >= new_width || pos_y >= new_height) continue;
        if (pos_x % scale == 0 && pos_y % scale == 0) d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b);
        else if (pos_x % scale == 0) d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_y % scale) * ((pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
        else if (pos_y % scale == 0) d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
        else{
            int x_y_r = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b)) / (scale));
            int x_y_1_r = pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale + 1, old_width, old_size, r_g_b) - pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b)) / (scale));
            d_new_image[pos_x + pos_y * new_width + new_size * r_g_b] = x_y_r + (pos_y % scale) * ((x_y_1_r - x_y_r) / scale);
        }
    }
}

__global__ void nearest_neighbor_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height, int pixel_per_thread, int imgs_size){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;
    float scale = (float)new_width / old_width;
    for (int i = 0; i < pixel_per_thread; i++){
        int pos = pixel_per_thread * (blockIdx.x * blockDim.x + threadIdx.x) + i;
        int img = pos /(new_size * 3);
        int img_pos = pos %(new_size * 3);
        int pos_x = (img_pos % new_size) % new_width;
        int pos_y = (img_pos % new_size) / new_width;
        int r_g_b = img_pos / new_size;
        if (img > imgs_size) break;
        d_new_image[img * new_size * 3 + pos_x + pos_y * new_width + new_size * r_g_b] = d_old_image[img * old_size * 3 + (int)(pos_x / scale) + (int)(pos_y / scale) * old_width + old_size * r_g_b];
    }
}

void interpolate(vector<string> paths, vector<string> file_names, int scale, int interpolation_mode){
    unsigned long long old_size = 0;
    unsigned long long new_size = 0;
    unsigned char *d_old_images;
    unsigned char *d_new_images;
    vector<CImg<unsigned char>> old_imgs;
    vector<CImg<unsigned char>> new_imgs;
    for (int i = 0; i < paths.size(); i++){
        old_imgs.push_back(CImg<unsigned char>(paths.at(i).c_str()));
        old_size += old_imgs.at(i).size();
    }

    for (int i = 0; i < paths.size(); i++){
        if(interpolation_mode == 1) new_imgs.push_back(CImg<unsigned char>(old_imgs.at(i).width() * scale, old_imgs.at(i).height() * scale, 1, 3, 255));
        if(interpolation_mode == 2) new_imgs.push_back(CImg<unsigned char>(old_imgs.at(i).width()* scale - (scale - 1), old_imgs.at(i).height() * scale - (scale - 1), 1, 3, 255));
        new_size += new_imgs.at(i).size();
    }
    cudaMalloc((void **)&d_old_images, old_size);
    cudaMalloc((void **)&d_new_images, new_size);

    for (int i = 0; i < paths.size(); i++){
        cudaMemcpy(d_old_images + i * old_imgs.at(i).size(), old_imgs.at(i).data(), old_imgs.at(i).size(), cudaMemcpyHostToDevice);
    }
    dim3 blkDim (block_size, 1, 1);
    dim3 grdDim ((((new_size/paths.size()) + block_size - 1)/block_size + pixel_per_thread - 1)/pixel_per_thread, 1, 1);

    if(interpolation_mode == 1) nearest_neighbor_interpolation<<<grdDim, blkDim>>>(d_old_images, d_new_images, old_imgs.at(0).width(), old_imgs.at(0).height(), new_imgs.at(0).width(), new_imgs.at(0).height(), pixel_per_thread, paths.size());
    if(interpolation_mode == 2) linear_interpolation<<<grdDim, blkDim>>>(d_old_images, d_new_images, old_imgs.at(0).width(), old_imgs.at(0).height(), new_imgs.at(0).width(), new_imgs.at(0).height(), scale, pixel_per_thread);

    cudaDeviceSynchronize();

    for (int i = 0; i < paths.size(); i++){
        cudaMemcpy(new_imgs.at(i).data(), d_new_images + i * new_imgs.at(0).size(), new_imgs.at(i).size(), cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < paths.size(); i++){
        cout << "Guardando " << file_names.at(i) << "..." << endl;
        string _ = "new_imgs/";
        _.append(file_names.at(i));
        new_imgs.at(i).save(_.c_str());
    }

    /*
    CImg<unsigned char> img_in(path.c_str());
    int old_size = img_in.size() / 3;
    int old_width = img_in.width();
    int old_height = img_in.height();
    unsigned long long size = img_in.size();
    unsigned char *old_image = img_in.data();

    CImg<unsigned char> img_out;
    if(interpolation_mode == 1) img_out = CImg<unsigned char>(old_width * scale, old_height * scale, 1, 3, 255);
    if(interpolation_mode == 2) img_out = CImg<unsigned char>(old_width * scale - (scale - 1), old_height * scale - (scale - 1), 1, 3, 255);
    unsigned char *new_image = img_out.data();
    int new_size = img_out.size() / 3;
    int new_width = img_out.width();
    int new_height = img_out.height();

    unsigned char *d_old_image;
    unsigned char *d_new_image;

    cudaMalloc((void **)&d_old_image, old_size * 3 * sizeof(unsigned char));
    cudaMalloc((void **)&d_new_image, new_size * 3 * sizeof(unsigned char));

    //TIMERSTART(parallel);
    cudaMemcpy(d_old_image, old_image, old_size * 3, cudaMemcpyHostToDevice);

    dim3 blkDim (block_size, 1, 1);
    dim3 grdDim ((((new_size * 3) + block_size - 1)/block_size + pixel_per_thread - 1)/pixel_per_thread, 1, 1);

    if(interpolation_mode == 1) nearest_neighbor_interpolation<<<grdDim, blkDim>>>(d_old_image, d_new_image, old_width, old_height, new_width, new_height, pixel_per_thread);
    if(interpolation_mode == 2) linear_interpolation<<<grdDim, blkDim>>>(d_old_image, d_new_image, old_width, old_height, new_width, new_height, scale, pixel_per_thread);

    cudaDeviceSynchronize();

    cudaMemcpy(new_image, d_new_image, new_size * 3, cudaMemcpyDeviceToHost);

    //TIMERSTOP(parallel);

    string _ = "new_imgs/";
    _.append(file_name);
    //cout << "saving " << file_name << "..." << endl;
    img_out.save(_.c_str());
    cudaFree(d_old_image);
    cudaFree(d_new_image);
    */
    cudaFree(d_old_images);
    cudaFree(d_new_images);
}

int main(int argc, char const *argv[]){
    int test = 0;
    int interpolation_mode = 0;
    int scale = 0;
    string path;
    if (argc < 4){
        cout << "Modo de uso: " << argv[0] << " \"Nombre imagen\" \"tecnica(NNI/LI)\" \"factor de escalado(ej: int >= 1)\"" << endl;
        return 1;
    }
    if(strcmp(argv[2], "NNI") == 0) interpolation_mode = 1;
    else if(strcmp(argv[2], "LI") == 0) interpolation_mode = 2;
    else{
        cout << "la tecnica de interpolacion puede ser \"NNI\" o \"LI\""<<endl;
        return 1;
    }
    scale = atoi(argv[3]);
    if(scale < 1){
        cout << "El factor de escalado debe ser un entero mayor o igual a 1" << endl;
        return 1;
    }
    if(argc > 4){
        if(strcmp(argv[4], "-t") == 0){
            cout <<"------------------ Test -------------------" << endl;
            test = 1;
        }
    }
    path = argv[1];
    TIMERSTART(ALL_IMGS);
    // leer todos los archivos de una carpeta
    vector<string> imgs;
    vector<string> names;
    if (auto dir = opendir(path.c_str())) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.') continue; // Skip everything that starts with a dot
            string _ = argv[1];
            if(_.at(_.length() - 1) != '/') _.append("/");
            _.append(f->d_name);
            imgs.push_back(_);
            names.push_back(f->d_name);
            if(imgs.size() == img_per_kernel){
                interpolate(imgs, names, scale, interpolation_mode);
                imgs.clear();
                names.clear();
            }
        }
        interpolate(imgs, names, scale, interpolation_mode);
        closedir(dir);
    }
    TIMERSTOP(ALL_IMGS);
    return 0;
}