//g++ sequential_linear_interpolation.cpp -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0 -o sequential_BLI
#include <iostream>
#include "CImg.h"
#include "./others/metrictime.hpp"

using namespace cimg_library;
using namespace std;

int pixel(unsigned char *img, int x, int y, int width, int size, int rgb){
    return img[(x) + (y) * width + size * rgb];
}

int scale;

int main(int argc, char const *argv[]){
    int test = 0;
    if(argc < 3){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: 2, 3)\""<<endl;
        return 1;
    }
    
    if(argc > 3){
        if(strcmp(argv[3], "test") == 0){
            cout <<"------------------ Test -------------------" << endl;
            test = 1;
        }
    }
    scale = atoi(argv[2]);
    if(scale < 1){
        cout << "El factor de escalado debe ser un entero mayor o igual a 1" << endl;
        return 1;
    }
    CImg<unsigned char> img_in(argv[1]);
    int old_size = img_in.size()/3;
    int old_width = img_in.width();
    int old_height = img_in.height();

    unsigned long long size = img_in.size();
    unsigned char *in = img_in.data();

    CImg<unsigned char> img_out(old_width*scale - (scale - 1), old_height*scale - (scale - 1), 1, 3, 255);
    unsigned char *out = img_out.data();
    int new_size = img_out.size()/3;
    int new_width = img_out.width();
    int new_height = img_out.height();

    TIMERSTART(SEQUENTIAL_BLI);
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
    TIMERSTOP(SEQUENTIAL_BLI);
    if(!test)img_out.save("new_img.jpg");
    return 0;
}
