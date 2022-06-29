//g++ sequential_linear_interpolation.cpp -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0
//g++ sequential_linear_interpolation.cpp -lX11
#include <iostream>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

int pixel(unsigned char *img,int x, int y, int rgb){

}

int scale;

int main(int argc, char const *argv[]){
    if(argc != 3){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\" \"factor de escalado(ej: int >= 1)\""<<endl;
        return 1;
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

    for (int y = 0; y < new_height; y++){
        for (int x = 0; x < new_width; x++){
            if(x%scale == 0 && y%scale == 0){       // se mantiene el color
                out[x + y*new_width] = in[(x/scale) + (y/scale) * old_width];
                out[x + y*new_width + new_size] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size];
                out[x + y*new_width + new_size*2] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2];
            }
            else if(x%scale == 0){                  // se interpola en el eje y
                out[x + y*new_width] = in[(int)(x/scale) + (int)(y/scale) * old_width] + (x%scale)*(in[(int)(x/scale) + ((int)(y/scale) + 1) * old_width] - in[(int)(x/scale) + (int)(y/scale) * old_width])/(scale);
                out[x + y*new_width + new_size] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size] + (x%scale)*(in[(int)(x/scale)  + ((int)(y/scale) + 1) * old_width + old_size] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size])/(scale);
                out[x + y*new_width + new_size*2] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2] + (x%scale)*(in[(int)(x/scale) + ((int)(y/scale) + 1) * old_width + old_size*2] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2])/(scale);
            }
            else if(y%scale == 0){                  // se interpola en el eje x
                out[x + y*new_width] = in[(int)(x/scale) + (int)(y/scale) * old_width] + (y%scale)*(in[(int)(y/scale) + 1 + (int)(y/scale) * old_width] - in[(int)(x/scale) + (int)(y/scale) * old_width])/(scale);
                out[x + y*new_width + new_size] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size] + (y%scale)*(in[(int)(x/scale) + 1 + (int)(y/scale) * old_width + old_size] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size])/(scale);
                out[x + y*new_width + new_size*2] = in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2] + (y%scale)*(in[(int)(x/scale) + 1 + (int)(y/scale) * old_width + old_size*2] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2])/(scale);
                
            }
            else{                                   // si
                //out[x + y*new_width] = (in[(int)(x/scale) + (int)(y/scale) * old_width] + (x%scale)*(in[(int)(x/scale) + ((int)(y/scale) + 1) * old_width] - in[(int)(x/scale) + (int)(y/scale) * old_width])/(scale-1) + in[(int)(x/scale) + (int)(y/scale) * old_width] + (y%scale)*(in[(int)(x/scale) + 1 + (int)(y/scale) * old_width] - in[(int)(x/scale) + (int)(y/scale) * old_width])/(scale-1))/2;
                //out[x + y*new_width + new_size] = (in[(int)(x/scale) + (int)(y/scale) * old_width + old_size] + (x%scale)*(in[(int)(x/scale)  + ((int)(y/scale) + 1) * old_width + old_size] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size])/(scale-1) + in[(int)(x/scale) + (int)(y/scale) * old_width + old_size] + (y%scale)*(in[(int)(x/scale) + 1 + (int)(y/scale) * old_width + old_size] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size])/(scale-1))/2;
                //out[x + y*new_width + new_size*2] = (in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2] + (x%scale)*(in[(int)(x/scale) + ((int)(y/scale) + 1) * old_width + old_size*2] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2])/(scale-1) + in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2] + (y%scale)*(in[(int)(x/scale) + 1 + (int)(y/scale) * old_width + old_size*2] - in[(int)(x/scale) + (int)(y/scale) * old_width + old_size*2])/(scale-1))/2;
            }
        }
    }
    img_out.save("new_img.jpg");
    return 0;
}
