#ifndef TIME_HELPER
#define TIME_HELPER

#include <iostream>
#include <cstdint>

#include <chrono>

#define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::high_resolution_clock> a##label, b##label; \
	a##label = std::chrono::high_resolution_clock::now();


#define TIMERSTOP(label)                                                   \
        b##label = std::chrono::high_resolution_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << "s" << std::endl;

#endif

