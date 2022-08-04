// nvcc parallel_video.cu -lX11
// nvcc parallel_video.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0

#include <iostream>
#include <vector>
#include <dirent.h>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

static int block_size = 1024;

static int pixel_per_thread = 128;

static int img_per_kernel = 128;

static int cpu_threads = 12;

int n_imgs = 0;

int imgs_ok = 0;

chrono::_V2::system_clock::time_point start;



__device__ int pixel(unsigned char *img, int x, int y, int width, int size, int rgb, int n_img, int spectrum){
    return img[n_img * size * spectrum + (x) + (y)*width + size * rgb];
}

__global__ void linear_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height, int  scale, int pixel_per_thread, int imgs_size, int spectrum){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;
    for (int i = 0; i < pixel_per_thread; i++){
        int pos = pixel_per_thread *(blockIdx.x * blockDim.x + threadIdx.x) + i;
        int img = pos /(new_size * spectrum);
        int img_pos = pos %(new_size * spectrum);
        int pos_x = (img_pos % new_size) % new_width;
        int pos_y = (img_pos % new_size) / new_width;
        int r_g_b = img_pos / new_size;
        if (img >= imgs_size) break;
        if (pos_x % scale == 0 && pos_y % scale == 0) d_new_image[img * new_size * spectrum + pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum);
        else if (pos_x % scale == 0) d_new_image[img * new_size * spectrum + pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum) + (pos_y % scale) * ((pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b, img, spectrum) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum)) / (scale));
        else if (pos_y % scale == 0) d_new_image[img * new_size * spectrum + pos_x + pos_y * new_width + new_size * r_g_b] = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b, img, spectrum) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum)) / (scale));
        else{
            int x_y_r = pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale, old_width, old_size, r_g_b, img, spectrum) - pixel(d_old_image, pos_x / scale, pos_y / scale, old_width, old_size, r_g_b, img, spectrum)) / (scale));
            int x_y_1_r = pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b, img, spectrum) + (pos_x % scale) * ((pixel(d_old_image, pos_x / scale + 1, pos_y / scale + 1, old_width, old_size, r_g_b, img, spectrum) - pixel(d_old_image, pos_x / scale, pos_y / scale + 1, old_width, old_size, r_g_b, img, spectrum)) / (scale));
            d_new_image[img * new_size * spectrum + pos_x + pos_y * new_width + new_size * r_g_b] = x_y_r + (pos_y % scale) * ((x_y_1_r - x_y_r) / scale);
        }
    }
}

__global__ void nearest_neighbor_interpolation(unsigned char *d_old_image, unsigned char *d_new_image, int old_width, int old_height, int new_width, int new_height, int pixel_per_thread, int imgs_size, int spectrum){
    int old_size = old_height * old_width;
    int new_size = new_height * new_width;
    float scale = (float)new_width / old_width;
    for (int i = 0; i < pixel_per_thread; i++){
        int pos = pixel_per_thread * (blockIdx.x * blockDim.x + threadIdx.x) + i;
        int img = pos /(new_size * spectrum);
        int img_pos = pos %(new_size * spectrum);
        int pos_x = (img_pos % new_size) % new_width;
        int pos_y = (img_pos % new_size) / new_width;
        int r_g_b = img_pos / new_size;
        if (img >= imgs_size) break;
        d_new_image[img * new_size * spectrum + pos_x + pos_y * new_width + new_size * r_g_b] = d_old_image[img * old_size * spectrum + (int)(pos_x / scale) + (int)(pos_y / scale) * old_width + old_size * r_g_b];
    }
}

void interpolate(vector<string> paths, vector<string> file_names, int scale, int interpolation_mode, bool test){
    cudaStream_t s_1;
    cudaStreamCreate(&s_1);
    unsigned long long old_size = 0;
    unsigned long long new_size = 0;
    unsigned char *d_old_images;
    unsigned char *d_new_images;
    vector<CImg<unsigned char>> old_imgs;
    vector<CImg<unsigned char>> new_imgs;
    
    for (int i = 0; i < paths.size(); i++){
        old_imgs.push_back(CImg<unsigned char>(paths.at(i).c_str()));
    }
    old_size = old_imgs.at(0).size() * old_imgs.size();

    for (int i = 0; i < paths.size(); i++){
        if(interpolation_mode == 1) new_imgs.push_back(CImg<unsigned char>(old_imgs.at(i).width() * scale, old_imgs.at(i).height() * scale, 1, old_imgs.at(0).spectrum(), 255));
        if(interpolation_mode == 2) new_imgs.push_back(CImg<unsigned char>(old_imgs.at(i).width()* scale - (scale - 1), old_imgs.at(i).height() * scale - (scale - 1), 1, old_imgs.at(0).spectrum(), 255));
    }
    new_size = new_imgs.at(0).size() * new_imgs.size();

    cudaMalloc((void **)&d_old_images, old_size);
    cudaMalloc((void **)&d_new_images, new_size);
    
    for (int i = 0; i < paths.size(); i++){
        cudaMemcpyAsync(d_old_images + i * old_imgs.at(i).size(), old_imgs.at(i).data(), old_imgs.at(i).size(), cudaMemcpyHostToDevice, s_1);
    }

    dim3 blkDim (block_size, 1, 1);
    dim3 grdDim ((new_size + block_size + pixel_per_thread - 1)/(block_size * pixel_per_thread), 1, 1);

    if(interpolation_mode == 1) nearest_neighbor_interpolation<<<grdDim, blkDim, 0, s_1>>>(d_old_images, d_new_images, old_imgs.at(0).width(), old_imgs.at(0).height(), new_imgs.at(0).width(), new_imgs.at(0).height(), pixel_per_thread, paths.size(), old_imgs.at(0).spectrum());
    if(interpolation_mode == 2) linear_interpolation<<<grdDim, blkDim, 0, s_1>>>(d_old_images, d_new_images, old_imgs.at(0).width(), old_imgs.at(0).height(), new_imgs.at(0).width(), new_imgs.at(0).height(), scale, pixel_per_thread, paths.size(), old_imgs.at(0).spectrum());

    cudaDeviceSynchronize();
    
    for (int i = 0; i < paths.size(); i++){
        cudaMemcpyAsync(new_imgs.at(i).data(), d_new_images + i * new_imgs.at(0).size(), new_imgs.at(i).size(), cudaMemcpyDeviceToHost, s_1);
    }
    
    cudaFree(d_old_images);

    if(!test){
        for (int i = 0; i < paths.size(); i++){
            string _ = "new_imgs/";
            _.append(file_names.at(i));
            new_imgs.at(i).save(_.c_str());
        }
    }

    cudaFree(d_new_images);
    cudaStreamDestroy(s_1);
}

int main(int argc, char const *argv[]){
    int test = 0;
    int interpolation_mode = 0;
    int scale = 0;
    thread threads[cpu_threads];
    string path;

    if (argc < 4){
        cout << "Modo de uso: " << argv[0] << " \"Nombre carpeta\" \"tecnica(NNI/LI)\" \"factor de escalado(ej: int >= 1)\"" << endl;
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
            test = 1;
        }
    }
    path = argv[1];
    // leer todos los archivos de una carpeta
    vector<vector<string>> imgs(cpu_threads);
    vector<vector<string>> names(cpu_threads);
    if (auto dir = opendir(path.c_str())) {
        while (auto f = readdir(dir)){
            if (!f->d_name || f->d_name[0] == '.') continue;
            n_imgs++;
        }
        closedir(dir);
    }
    system("clear");
    if(test) cout << "[TEST] ";
    cout << "Total: "<< n_imgs << " imagenes a procesar..."<<endl;
    // leer todos los archivos de una carpeta
    if (auto dir = opendir(path.c_str())) {
        start = chrono::system_clock::now();
        int actual_thread = 0;
        while (auto f = readdir(dir)) {
            if(threads[actual_thread].joinable()){
                for (int i = 0; i < cpu_threads; i++){
                    threads[i].join();
                    vector<string>().swap(imgs.at(i));
                    vector<string>().swap(names.at(i));
                }
                system("clear");
                if(test) cout << "[TEST] ";
                imgs_ok += cpu_threads * img_per_kernel;
                chrono::duration<float,milli> duration = chrono::system_clock::now() - start;
                cout << "["<< imgs_ok <<"/"<< n_imgs<< "] " << ((float)imgs_ok/n_imgs)*100 << "% \nTiempo restante "<< (duration.count()/1000)/((float)imgs_ok/n_imgs) - duration.count()/1000 <<"s"<< endl;                 
            }
            
            if (!f->d_name || f->d_name[0] == '.') continue; // Skip everything that starts with a dot
            string _ = argv[1];
            if(_.at(_.length() - 1) != '/') _.append("/");
            _.append(f->d_name);

            imgs.at(actual_thread).push_back(_);
            names.at(actual_thread).push_back(f->d_name);
            if(imgs.at(actual_thread).size() == img_per_kernel){
                threads[actual_thread] = thread(interpolate, imgs.at(actual_thread), names.at(actual_thread), scale, interpolation_mode, test);
                actual_thread++;
                if(actual_thread == cpu_threads) actual_thread = 0;
            }
        }
        for (int i = 0; i < cpu_threads; i++) if(threads[i].joinable()) threads[i].join();
        if(imgs.at(actual_thread).size() != 0 && imgs.at(actual_thread).size() != img_per_kernel) interpolate(imgs.at(actual_thread), names.at(actual_thread), scale, interpolation_mode, test);
        chrono::duration<float,milli> duration = chrono::system_clock::now() - start;
        system("clear");
        cout << n_imgs <<" imagenes procesadas en "<< duration.count()/1000 <<"s"<< endl;
        closedir(dir);
    }
    return 0;
}