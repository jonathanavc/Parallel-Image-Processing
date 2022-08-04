// g++ sequential_video.cpp -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0

#include <iostream>
#include <vector>
#include <dirent.h>
#include <chrono>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

int n_imgs = 0;
int imgs_ok = 0;

chrono::_V2::system_clock::time_point start;

int pixel(unsigned char *img, int x, int y, int width, int size, int rgb){
    return img[size * 3 + (x) + (y)*width + size * rgb];
}

void linear_interpolation(unsigned char *in, unsigned char *out, int old_width, int old_height, int new_width, int new_height,int old_size,int new_size, int  scale){
    for (int y = 0; y < new_height; y++){
        for (int x = 0; x < new_width; x++){
            if(x%scale == 0 && y%scale == 0){       // se mantiene el color ✅
                out[x + y*new_width] = pixel(in, x/scale, y/scale,old_width,old_size, 0);
                out[x + y*new_width + new_size] = pixel(in, x/scale, y/scale,old_width,old_size, 1);
                out[x + y*new_width + new_size*2] = pixel(in, x/scale, y/scale,old_width,old_size, 2);
            }
            else if(x%scale == 0){                  // se interpola en el eje y ✅
                out[x + y*new_width] = pixel(in, x/scale, y/scale,old_width,old_size, 0) + (y%scale)*((pixel(in, x/scale, y/scale + 1, old_width, old_size, 0) - pixel(in, x/scale, y/scale, old_width,old_size, 0))/(scale));
                out[x + y*new_width + new_size] = pixel(in, x/scale, y/scale,old_width,old_size, 1) + (y%scale)*((pixel(in, x/scale, y/scale + 1, old_width, old_size, 1) - pixel(in, x/scale, y/scale, old_width,old_size, 1))/(scale));
                out[x + y*new_width + new_size*2] = pixel(in, x/scale, y/scale,old_width,old_size, 2) + (y%scale)*((pixel(in, x/scale, y/scale + 1, old_width, old_size, 2) - pixel(in, x/scale, y/scale, old_width,old_size, 2))/(scale));
            }
            else if(y%scale == 0){                  // se interpola en el eje x ✅
                out[x + y*new_width] = pixel(in, x/scale, y/scale,old_width,old_size, 0) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 0) - pixel(in, x/scale, y/scale, old_width,old_size, 0))/(scale));
                out[x + y*new_width + new_size] = pixel(in, x/scale, y/scale,old_width,old_size, 1) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 1) - pixel(in, x/scale, y/scale, old_width,old_size, 1))/(scale));
                out[x + y*new_width + new_size*2] = pixel(in, x/scale, y/scale,old_width,old_size, 2) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 2) - pixel(in, x/scale, y/scale, old_width,old_size, 2))/(scale));
            }
            else{                                   // ahora si (creo)✅✅✅✅✅🚬🐛
                // obtener (x,y)
                int x_y_r = pixel(in, x/scale, y/scale,old_width,old_size, 0) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 0) - pixel(in, x/scale, y/scale, old_width,old_size, 0))/(scale));
                int x_y_g = pixel(in, x/scale, y/scale,old_width,old_size, 1) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 1) - pixel(in, x/scale, y/scale, old_width,old_size, 1))/(scale));
                int x_y_b = pixel(in, x/scale, y/scale,old_width,old_size, 2) + (x%scale)*((pixel(in, x/scale + 1, y/scale, old_width, old_size, 2) - pixel(in, x/scale, y/scale, old_width,old_size, 2))/(scale));
                // obtener (x,y+1)
                int x_y_1_r = pixel(in, x/scale, y/scale + 1,old_width,old_size, 0) + (x%scale)*((pixel(in, x/scale + 1, y/scale + 1 , old_width, old_size, 0) - pixel(in, x/scale, y/scale + 1, old_width,old_size, 0))/(scale));
                int x_y_1_g = pixel(in, x/scale, y/scale + 1,old_width,old_size, 1) + (x%scale)*((pixel(in, x/scale + 1, y/scale + 1 , old_width, old_size, 1) - pixel(in, x/scale, y/scale + 1, old_width,old_size, 1))/(scale));
                int x_y_1_b = pixel(in, x/scale, y/scale + 1,old_width,old_size, 2) + (x%scale)*((pixel(in, x/scale + 1, y/scale + 1 , old_width, old_size, 2) - pixel(in, x/scale, y/scale + 1, old_width,old_size, 2))/(scale));
                //interpolar
                out[x + y*new_width] = x_y_r + (y%scale)*((x_y_1_r - x_y_r)/scale);
                out[x + y*new_width + new_size] = x_y_g + (y%scale)*((x_y_1_g - x_y_g)/scale);
                out[x + y*new_width + new_size*2] = x_y_b + (y%scale)*((x_y_1_b - x_y_b)/scale);
            }
        }
    }
}

void nearest_neighbor_interpolation(unsigned char *in, unsigned char *out, int old_width, int old_height, int new_width, int new_height, int old_size, int new_size, int scale){
    for (int y = 0; y < new_height; y++){
        for (int x = 0; x < new_width; x++){
            out[x + y*new_width] = in[(int)(x/scale) + (int)(y/scale) * old_width];
            out[x + y*new_width + new_size] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size];
            out[x + y*new_width + new_size*2] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2];
        }
    }
}

int main(int argc, char const *argv[]){
    int test = 0;
    int interpolation_mode = 0;
    int scale = 0;
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
        start = std::chrono::system_clock::now();
        while (auto f = readdir(dir)) {

            if (!f->d_name || f->d_name[0] == '.') continue;

            string _ = argv[1];
            if(_.at(_.length() - 1) != '/') _.append("/");
            _.append(f->d_name);

            CImg<unsigned char> img_in(_.c_str());
            int old_size = img_in.size()/3;
            int old_width = img_in.width();
            int old_height = img_in.height();

            unsigned long long size = img_in.size();
            unsigned char *in = img_in.data();

            CImg<unsigned char> img_out;
            if(interpolation_mode == 1) img_out = CImg<unsigned char>(old_width*scale, old_height*scale, 1, 3, 255);
            if(interpolation_mode == 2) img_out = CImg<unsigned char>(old_width*scale - (scale - 1), old_height*scale - (scale - 1), 1, 3, 255);
            unsigned char *out = img_out.data();
            int new_size = img_out.size()/3;
            int new_width = img_out.width();
            int new_height = img_out.height();
            
            if(interpolation_mode == 1) nearest_neighbor_interpolation(in, out, old_width, old_height, new_width, new_height, old_size, new_size, scale);
            if(interpolation_mode == 2) linear_interpolation(in, out, old_width, old_height, new_width, new_height, old_size, new_size,scale);

            if(!test){
                string _ = "new_imgs/";
                _.append(f->d_name);
                img_out.save(_.c_str());
            }
            system("clear");
            if(test) cout << "[TEST] ";
            imgs_ok ++;
            std::chrono::duration<float,std::milli> duration = std::chrono::system_clock::now() - start;
            cout << "["<< imgs_ok <<"/"<< n_imgs<< "] " << ((float)imgs_ok/n_imgs)*100 << "% \nTiempo restante "<< (duration.count()/1000)/((float)imgs_ok/n_imgs) - duration.count()/1000 <<"s"<< endl;
        }
        std::chrono::duration<float,std::milli> duration = std::chrono::system_clock::now() - start;
        system("clear");
        cout << n_imgs <<" imagenes procesadas en "<< duration.count()/1000 <<"s"<< endl;
        closedir(dir);
    }
    return 0;
}