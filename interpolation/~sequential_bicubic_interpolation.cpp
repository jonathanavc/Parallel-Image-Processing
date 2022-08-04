//g++ sequential_linear_interpolation.cpp -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0 -o sequential_BLI
#include <iostream>
#include "CImg.h"
#include <algorithm>  
#include "./others/metrictime.hpp"

using namespace cimg_library;
using namespace std;

int scale;
/*
double cubicInterpolate (double p[4], double x) {
    return p[1] + 0.5 * x*(p[2] - p[0] + x*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + x*(3.0*(p[1] - p[2]) + p[3] - p[0])));
}
*/
double cubicInterpolate(double p[4], double x){
    double A = (p[3]-p[2])-(p[0]-p[1]);
    double B = (p[0]-p[1])-A;
    double C = p[2]-p[0];
    double D = p[1];
    return D + x * (C + x * (B + x * A));
}

double bicubicInterpolate (double p[4][4], double x, double y) {
    double arr[4];
    arr[0] = cubicInterpolate(p[0], y);
    arr[1] = cubicInterpolate(p[1], y);
    arr[2] = cubicInterpolate(p[2], y);
    arr[3] = cubicInterpolate(p[3], y);
    return cubicInterpolate(arr, x);
}

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
    for (int y = 0; y < new_height/(4 * scale); y++){
        for (int x = 0; x < new_width/(4 * scale); x++){
            double array[4][4];
            for (int r_g_b = 0; r_g_b < 3; r_g_b++){
                for (int ii = 0; ii < 4; ii++){
                    for (int jj = 0; jj < 4; jj++){
                        array[ii][jj] = in[r_g_b * old_size + (y * 4 + ii) * old_width + x * 4 + jj];
                    }
                }
                for (int ii = 0; ii < 4 * scale; ii++){
                    for (int jj = 0; jj < 4 * scale; jj++){
                        out[r_g_b * new_size + (y*scale*4 + ii) * new_width + (x*4*scale + jj)] = max(0,min((int)bicubicInterpolate(array, ii/(scale), jj/(scale)), 255));
                    }
                }
            }
        }
    }
    TIMERSTOP(SEQUENTIAL_BLI);
    if(!test)img_out.save("new_img.jpg");
    return 0;
}