//nvcc parallel_cimg.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0
#include <iostream>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

int main(int argc, char const *argv[]){
    if(argc != 2){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\""<<endl;
        return 1;
    }
    CImg<unsigned char> img_in(argv[1]);
    int width = img_in.width();
    int height = img_in.height();

    unsigned long long size = img_in.size();
    unsigned char *in = img_in.data();

    CImg<unsigned char> img_out(width, height, 1, 3, 255);
    unsigned char *out = img_out.data();
    int img_size = img_out.size()/3;
    for (int i = 0; i < img_size; i++){
        //R
        out[i] = min(in[i]+50,255);
        //G           
        out[i + img_size] = in[i + img_size];
        //B          
        out[i + img_size*2] = in[i + img_size*2];
    }
    img_out.save("new_img.jpg");
    return 0;
}
