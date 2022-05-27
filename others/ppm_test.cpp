//g++ sequential.cpp -std=c++11
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;


int main(int argc, char const *argv[]){
    srand(time(NULL));
    ifstream in_image;
    ofstream out_image;
    if(argc != 2){
        cout << "Modo de uso: "<< argv[0] << " \"Nombre_imagen\""<<endl;
        return 1;
    }
    in_image.open(argv[1]);
    out_image.open("new_img.ppm");
    string s;
    in_image >> s;                  //type
    out_image << s << endl;
    in_image >> s;                  //width
    out_image << s <<" ";
    in_image >> s;                  //height
    out_image << s << endl;
    in_image >> s;                  //RGB
    out_image << s << endl;
    string red;
    string green;
    string blue;
    int r;
    int g;
    int b;
    while (!in_image.eof()){
        in_image >> red;
        in_image >> green;
        in_image >> blue;
        r = stoi(red);
        g = stoi(green);
        b = stoi(blue);

        out_image << min(r + 50,255) << "\n";
        out_image << g << "\n";
        out_image << b << "\n";
    }
    return 0;
}
