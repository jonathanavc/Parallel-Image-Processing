//nvcc parallel_cimg.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0
#include <iostream>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

float scale;

int main(int argc, char const *argv[]){
    if(argc != 3){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: 2, 1.5)\""<<endl;
        return 1;
    }
    scale = atof(argv[2]);
    CImg<unsigned char> img_in(argv[1]);
    int old_size = img_in.size()/3;
    int old_width = img_in.width();
    int old_height = img_in.height();

    unsigned long long size = img_in.size();
    unsigned char *in = img_in.data();

    CImg<unsigned char> img_out(old_width*scale, old_height*scale, 1, 3, 255);
    unsigned char *out = img_out.data();
    int new_size = img_out.size()/3;
    int new_width = img_out.width();
    int new_height = img_out.height();

    for (int y = 0; y < new_height; y++){
        for (int x = 0; x < new_width; x++){
            //R
            out[x + y*new_width] = in[(int)(x/scale) + (int)(y/scale) * old_width];
            //G
            out[x + y*new_width + new_size] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size];
            //B
            out[x + y*new_width + new_size*2] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2];
        }
    }
    img_out.save("new_img.jpg");
    return 0;
}

