//g++ -std=c++11 -O3 -march=native -Dcimg_jpeg=1 -Dcimg_display=0  jpeg_to_ppm.cpp
#include <iostream>
#include "CImg.h"

using namespace cimg_library;

int main(int argc, char const *argv[]){
    if(argc != 2){
        std::cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\""<<std::endl;
        return 1;
    }
    CImg<unsigned char> img(argv[1]);
    img.save("ppm_image.ppm");
    return 0;
}
