#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <map>

#include "object.h"

struct Light{
    vec3 pos;
    vec3 color;
    float intensity;

    Light(const vec3 &pos, const vec3 &color, float intensity) : pos(pos), color(color), intensity(intensity) {}
};

struct DevLight{
    vec3 pos;
    vec3 color;
    float intensity;
};

class Camera;

class Scene{
    // host rendering
    std::vector<Object> objects;
    std::vector<Light> lights;

    // gpu rendering
    int n_objects, n_lights;
    
    DevObject *dev_objects;
    DevLight *dev_lights;
    
    bool gpu_render_ready;

    // helper
    std::vector<void*> dev_resources;
    friend class Camera;
public:
    Scene(){
        n_objects = 0;
        n_lights = 0;

        dev_objects = nullptr;
        dev_lights = nullptr;

        gpu_render_ready = true;
    }
    ~Scene(){
        if(gpu_render_ready){
            for(size_t i = 0; i < dev_resources.size(); ++i){
                CHECK_CUDA_CALL_ERROR(cudaFree(dev_resources[i]));
            }
        }
    }

    void add_object(const Object &object){
        objects.push_back(object);
        gpu_render_ready = false;
    }

    void add_light(const Light &light){
        lights.push_back(light);
        gpu_render_ready = false;
    }

    void prepare_gpu() {
        dev_resources.clear();

        n_objects = objects.size();
        n_lights = lights.size();
    
        DevObject *dev_objects_local = new DevObject[n_objects];
        
        std::map<Texture*, DevTexture*> m;
        std::map<Texture*, DevTexture*>::iterator it;
        for(int i = 0; i < n_objects; ++i){
            if(objects[i].texture != nullptr){
                if((it = m.find(objects[i].texture)) == m.end()){
                    DevTexture tex;
                    tex.w = objects[i].texture->width();
                    tex.h = objects[i].texture->height();
                    
                    CHECK_CUDA_CALL_ERROR(cudaMalloc(&tex.data, tex.w * tex.h * sizeof(uchar4)));
                    dev_resources.push_back((void*)tex.data);

                    CHECK_CUDA_CALL_ERROR(cudaMemcpy(tex.data, objects[i].texture->data_ptr(), tex.w * tex.h * sizeof(uchar4), cudaMemcpyHostToDevice));
    
                    DevTexture *dev_tex;
                    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_tex, sizeof(DevTexture)));

                    dev_resources.push_back((void*)dev_tex);

                    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_tex, &tex, sizeof(DevTexture), cudaMemcpyHostToDevice));
    
                    m[objects[i].texture] = dev_tex;

                    dev_objects_local[i].texture = dev_tex;
                }
                else{
                    dev_objects_local[i].texture = it->second;
                }   
            }
            else{
                dev_objects_local[i].texture = nullptr;
            }
        }

        for(int i = 0; i < n_objects; ++i){
            dev_objects_local[i].n_vertices = objects[i].vertices.size();
            dev_objects_local[i].n_normals = objects[i].normals.size();
            dev_objects_local[i].n_texture_coords = objects[i].texture_coords.size();
            dev_objects_local[i].n_trigs = objects[i].trigs.size();

            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_objects_local[i].vertices, dev_objects_local[i].n_vertices * sizeof(vec3)));
            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_objects_local[i].normals, dev_objects_local[i].n_normals * sizeof(vec3)));
            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_objects_local[i].texture_coords, dev_objects_local[i].n_texture_coords * sizeof(vec3)));
            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_objects_local[i].trigs, dev_objects_local[i].n_trigs * sizeof(Triangle)));
            
            dev_resources.push_back((void*)dev_objects_local[i].vertices);
            dev_resources.push_back((void*)dev_objects_local[i].normals);
            dev_resources.push_back((void*)dev_objects_local[i].texture_coords);
            dev_resources.push_back((void*)dev_objects_local[i].trigs);

            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_objects_local[i].vertices, objects[i].vertices.data(), dev_objects_local[i].n_vertices * sizeof(vec3), cudaMemcpyHostToDevice));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_objects_local[i].normals, objects[i].normals.data(), dev_objects_local[i].n_normals * sizeof(vec3), cudaMemcpyHostToDevice));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_objects_local[i].texture_coords, objects[i].texture_coords.data(), dev_objects_local[i].n_texture_coords * sizeof(vec3), cudaMemcpyHostToDevice));
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_objects_local[i].trigs, objects[i].trigs.data(), dev_objects_local[i].n_trigs * sizeof(Triangle), cudaMemcpyHostToDevice));

            dev_objects_local[i].refractive_index = objects[i].refractive_index;
            dev_objects_local[i].ambient_color = objects[i].ambient_color;
            dev_objects_local[i].diffuse_color = objects[i].diffuse_color;
            dev_objects_local[i].specular_color = objects[i].specular_color;
            dev_objects_local[i].ka = objects[i].ka;
            dev_objects_local[i].kd = objects[i].kd;
            dev_objects_local[i].ks = objects[i].ks;
            dev_objects_local[i].shininess = objects[i].shininess;
            dev_objects_local[i].kt = objects[i].kt;
            dev_objects_local[i].kr = objects[i].kr;
        }
    
        CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_objects, n_objects * sizeof(DevObject)));
        dev_resources.push_back((void*)dev_objects);

        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_objects, dev_objects_local, n_objects * sizeof(DevObject), cudaMemcpyHostToDevice));
    
        DevLight *dev_lights_local = new DevLight[n_lights];
        for(int i = 0; i < n_lights; ++i){
            dev_lights_local[i].pos = lights[i].pos;
            dev_lights_local[i].color = lights[i].color;
            dev_lights_local[i].intensity = lights[i].intensity;
        }
    
        CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_lights, n_lights * sizeof(DevLight)));
        dev_resources.push_back((void*)dev_lights);

        CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_lights, dev_lights_local, n_lights * sizeof(DevLight), cudaMemcpyHostToDevice));
    
        delete[] dev_objects_local;
        delete[] dev_lights_local;

        gpu_render_ready = true;
    }

    bool gpu_ready(){
        return gpu_render_ready;
    }
};

#endif