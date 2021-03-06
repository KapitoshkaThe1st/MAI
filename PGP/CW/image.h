#ifndef IMAGE_H
#define IMAGE_H

#include <fstream>
#include <string>

#include "utils.h"
#include "vec.h"

void save_ppm(vec3 *buffer, int w, int h, const std::string &path){
    std::ofstream ofs(path, std::ios::out | std::ios::binary); 
    ofs << "P6\n" << w << " " << h << "\n255\n"; 
    for (int i = 0; i < h * w; ++i) { 
        unsigned char r = (unsigned char)(255.0f * clamp(0.0f, buffer[i].x, 1.0f)); 
        unsigned char g = (unsigned char)(255.0f * clamp(0.0f, buffer[i].y, 1.0f)); 
        unsigned char b = (unsigned char)(255.0f * clamp(0.0f, buffer[i].z, 1.0f));
        ofs << r << g << b; 
    }
    ofs.close();
}

void save_data(vec3 *buffer, int w, int h, const std::string &path){
    std::ofstream ofs(path, std::ios::out | std::ios::binary); 
    ofs.write((char*)&w, sizeof(int)); 
    ofs.write((char*)&h, sizeof(int)); 
    for (int i = 0; i < h * w; ++i) { 
        unsigned char r = (unsigned char)(255.0f * clamp(0.0f, buffer[i].x, 1.0f)); 
        unsigned char g = (unsigned char)(255.0f * clamp(0.0f, buffer[i].y, 1.0f)); 
        unsigned char b = (unsigned char)(255.0f * clamp(0.0f, buffer[i].z, 1.0f)); 
        char a = 0; 
        ofs.write((char*)&r, sizeof(unsigned char));  
        ofs.write((char*)&g, sizeof(unsigned char));  
        ofs.write((char*)&b, sizeof(unsigned char));  
        ofs.write((char*)&a, sizeof(unsigned char));  
    }
    ofs.close();
}

#endif