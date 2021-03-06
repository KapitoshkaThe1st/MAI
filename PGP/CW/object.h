#ifndef OBJECT_H
#define OBJECT_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

#include "vec.h"
#include "mat.h"

struct Triangle{
    int v1, v2, v3;
    int n1, n2, n3;
    int t1, t2, t3;
    Triangle(int v1, int v2, int v3, int n1, int n2, int n3, int t1, int t2, int t3)
     : v1(v1), v2(v2), v3(v3), n1(n1), n2(n2), n3(n3), t1(t1), t2(t2), t3(t3) {}
};

enum class MaterialType{
    OPAQUE, TRANSPARENT, REFLECTIVE
};

class Texture{
    std::vector<uchar4> data;
    int w, h;

public:
    Texture() : w(0), h(0) {}
    static Texture import_ppm(std::string path){
        Texture res;

        std::ifstream ifs(path, std::ios::binary);
        if(!ifs){
            std::cout << "Can't open " << path << std::endl;
            exit(0);
        }
        std::string trash;
        ifs >> trash;
        int max_col_val;
        ifs >> res.w >> res.h >> max_col_val;
        ifs.get();
        size_t size = res.w * res.h;
        res.data.resize(size);
        for(size_t i = 0; i < size; ++i){
            unsigned char color[3];
            ifs.read((char*)color, 3 * sizeof(unsigned char));

            res.data[i] = make_uchar4(color[0], color[1], color[2], 0);
        }

        return res;
    }
    static Texture import_data(std::string path){
        std::ifstream ifs(path, std::ios::binary);
        if(!ifs){
            std::cout << "Can't open " << path << std::endl;
            exit(0);
        }
        Texture tex;
        ifs.read((char*)&tex.w, sizeof(int));
        ifs.read((char*)&tex.h, sizeof(int));

        tex.data.resize(tex.w * tex.h);
        ifs.read((char*)tex.data.data(), tex.w * tex.h * sizeof(uchar4));
        ifs.close();
        return tex;
    }
    
    uchar4 get_color(const vec3 &pos) const {
        if(w == 0 || h == 0){
            return make_uchar4(0, 0, 0, 0);
        }
        int x = pos.x * w;
        int y = pos.y * h;

        return data[y * w + x];
    }
    int width(){
        return w;
    }

    int height(){
        return h;
    }

    uchar4* data_ptr() {
        return data.data();
    }

    void change_color(const vec3 &color){
        for(int i = 0; i < h; ++i){
            for(int j = 0; j < w; ++j){
                data[i * w + j].x = (unsigned char)(color.x * data[i * w + j].x);
                data[i * w + j].y = (unsigned char)(color.y * data[i * w + j].y);
                data[i * w + j].z = (unsigned char)(color.z * data[i * w + j].z);
            }
        }
    }
};

struct Object{
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec3> texture_coords;
    std::vector<Triangle> trigs;

    float refractive_index;

    MaterialType material_type;

    vec3 ambient_color;
    vec3 diffuse_color;
    vec3 specular_color;

    Texture *texture;

    float ka;
    float kd;
    float ks;
    float shininess;

    float kt;
    float kr;

    bool has_texture() const {
        return texture != nullptr;
    }

    Object() : ambient_color(vec3(1.0f)), diffuse_color(vec3(1.0f)), specular_color(vec3(1.0f)),
               texture(nullptr),
               ka(0.2f), kd(0.6), ks(0.5), shininess(64.0f) { }

    void assign_texture(Texture *texture){
        this->texture = texture;
    }

    Object& translate(const vec3 &pos){
        for(size_t i = 0; i < vertices.size(); ++i)
            vertices[i] += pos;
        return *this;
        
    }

    Object& rotate(const vec3 &rot){
        float sx = sin(rot.x), sy = sin(rot.y), sz = sin(rot.z);
        float cx = cos(rot.x), cy = cos(rot.y), cz = cos(rot.z);
        mat3 mx(1.0f, 0.0f, 0.0f,
                0.0f, cx, -sx,
                0.0f, sx, cx);
        mat3 my(cy, 0.0f, sy,
                0.0f, 1.0f, 0.0f,
                -sy, 0.0f, cy);
        mat3 mz(cz, -sz, 0.0f,
                sz, cz, 0.0f,
                0.0f, 0.0f, 1.0f);
        mat3 r = (mx * my * mz);
        for(size_t i = 0; i < vertices.size(); ++i){
            vertices[i] =  r * vertices[i];
        }

        for(size_t i = 0; i < normals.size(); ++i){
            normals[i] =  r * normals[i];
        }

        return *this;
    }

    Object& rotate(const mat3 &m){
        for(size_t i = 0; i < vertices.size(); ++i){
            vertices[i] =  m * vertices[i];
        }

        for(size_t i = 0; i < normals.size(); ++i){
            normals[i] =  m * normals[i];
        }

        return *this;
    }

    Object& scale(const vec3 &scale){
        for(size_t i = 0; i < vertices.size(); ++i){
            vertices[i].x *= scale.x;
            vertices[i].y *= scale.y;
            vertices[i].z *= scale.z;   
        }
        return *this;
    }
};

std::vector<std::string> split(const std::string &str, char del = ' '){
    size_t len = str.length();
    size_t ind = 0, pos = 0;
    std::vector<std::string> res;
    while(ind < len){
        pos = str.find(del, ind);
        if(pos == std::string::npos){
            // res.push_back(str.substr(ind, string::npos)); // until the end of the string
            res.push_back(str.substr(ind, len - ind)); // until the end of the string
            break;
        }
        res.push_back(str.substr(ind, pos - ind));
        ind = pos + 1;
    }
    return res;
}

Object import_obj(const std::string &path){
    Object res;

    std::ifstream ifs(path);
    if(!ifs){
        std::cout << "Can't open " << path << std::endl;
        exit(0);
    }
    std::string line;
    while(std::getline(ifs, line)){
        std::vector<std::string> tmp = split(line);

        // for(auto s : tmp)
        //     std::cout << s << ' ';
        // std::cout << std::endl;

        if(line.empty()){
            continue;
        }
        else if(tmp[0] == "v"){

            float x = std::stof(tmp[2]);
            float y = std::stof(tmp[3]);
            float z = std::stof(tmp[4]);

            res.vertices.push_back(vec3(x, y, z));
        }
        else if(tmp[0] == "vt"){
            float u, v, w = 0.0f;
            u = std::stof(tmp[1]);
            v = std::stof(tmp[2]);
            if(tmp.size() > 3)
                w = std::stof(tmp[3]);

            res.texture_coords.push_back(vec3(u, v, w));
        }
        else if(tmp[0] == "vn"){
            float x = std::stof(tmp[1]);
            float y = std::stof(tmp[2]);
            float z = std::stof(tmp[3]);

            res.normals.push_back(vec3(x, y, z));
        }
        else if(tmp[0] == "f"){
            std::vector<std::string> nums = split(tmp[1], '/');
            int v1 = std::stoi(nums[0]) - 1;
            int t1 = std::stoi(nums[1]) - 1;
            int n1 = std::stoi(nums[2]) - 1;
            nums = split(tmp[2], '/');
            int v2 = std::stoi(nums[0]) - 1;
            int t2 = std::stoi(nums[1]) - 1;
            int n2 = std::stoi(nums[2]) - 1;
            nums = split(tmp[3], '/');
            int v3 = std::stoi(nums[0]) - 1;
            int t3 = std::stoi(nums[1]) - 1;
            int n3 = std::stoi(nums[2]) - 1;

            res.trigs.push_back(Triangle(v1, v2, v3, n1, n2, n3, t1, t2, t3));
        }
    }

    return res;
}

struct DevTexture{
    uchar4 *data;
    int w, h;

    __device__ uchar4 get_color(const vec3 &pos) const {
        if(w == 0 || h == 0){
            return make_uchar4(0, 0, 0, 0);
        }
        int x = min(w, max((int)(pos.x * w), 0));
        int y = min(h, max((int)(pos.y * h), 0));

        return data[y * w + x];
    }
};

struct DevObject{
    vec3 *vertices;
    vec3 *normals;
    vec3 *texture_coords;
    Triangle *trigs;

    int n_vertices;
    int n_normals;
    int n_texture_coords;
    int n_trigs;

    float refractive_index;

    vec3 ambient_color;
    vec3 diffuse_color;
    vec3 specular_color;

    DevTexture *texture;

    float ka;
    float kd;
    float ks;
    float shininess;

    float kt;
    float kr;

    __device__ bool has_texture() const {
        return texture != nullptr;
    }
};

#endif