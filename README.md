# Parallel-Image-Processing

**requirements:**
- sudo apt-get install -y nvidia-cuda-toolkit
- sudo apt-get install -y cimg-dev
- sudo apt-get install -y imagemagick

**compile**
- (cuda) nvcc parallel_video.cu -lX11
- (cuda) nvcc parallel_video.cu -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0 
- (c++) g++ sequential_video.cpp -std=c++11 -O3 -Dcimg_jpeg=1 -Dcimg_display=0