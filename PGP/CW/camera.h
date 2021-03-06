#ifndef CAMERA_H
#define CAMERA_H

#include "vec.h"
#include "mat.h"
#include "error.h"
#include "render.h"
#include "scene.h"

struct RenderOptions{
    int w, h;
    float fov;
    int max_depth;
    vec3 background_color;
};

class Camera{
    int last_w, last_h;
    int *dev_ray_counts;
    int ray_count;

    vec3 position;
    vec3 target_position;

    vec3 *dev_frame_buffer;

    mat4 lookat_mat;
    void build_lookat_mat(const vec3 &tmp = vec3(0.0f, 0.0f, 1.0f)){
    // void build_lookat_mat(const vec3 &tmp = vec3(0.0f, 1.0f, 0.0f)){
        vec3 forward = normalize(position - target_position); 
        vec3 right = cross_product(normalize(tmp), forward); 
        vec3 up = cross_product(forward, right);

        lookat_mat.m[0][0] = right.x;
        lookat_mat.m[1][0] = right.y;
        lookat_mat.m[2][0] = right.z;
        lookat_mat.m[3][0] = 0.0f;

        lookat_mat.m[0][1] = up.x;
        lookat_mat.m[1][1] = up.y;
        lookat_mat.m[2][1] = up.z;
        lookat_mat.m[3][1] = 0.0f;

        lookat_mat.m[0][2] = forward.x;
        lookat_mat.m[1][2] = forward.y;
        lookat_mat.m[2][2] = forward.z;
        lookat_mat.m[3][2] = 0.0f;

        lookat_mat.m[0][3] = position.x;
        lookat_mat.m[1][3] = position.y;
        lookat_mat.m[2][3] = position.z;
        lookat_mat.m[3][3] = 1.0f;
    }

public:
    Camera(const vec3 &pos, const vec3 &target_pos) : position(pos), target_position(target_pos) {
        build_lookat_mat();
        dev_frame_buffer = nullptr;
        dev_ray_counts = nullptr;
        ray_count = 0;
    }
    ~Camera(){
        if(dev_ray_counts != nullptr && dev_frame_buffer != nullptr){
            CHECK_CUDA_CALL_ERROR(cudaFree(dev_frame_buffer));
            CHECK_CUDA_CALL_ERROR(cudaFree(dev_ray_counts));
        }
    }

    int get_ray_count(){
        if(dev_ray_counts != nullptr && dev_frame_buffer != nullptr){
            int *ray_counts = new int[last_w * last_h];
            CHECK_CUDA_CALL_ERROR(cudaMemcpy(ray_counts, dev_ray_counts, last_w * last_h * sizeof(int), cudaMemcpyDeviceToHost));
            for(int i = 0; i < last_h; ++i){
                for(int j = 0; j < last_w; ++j){
                    ray_count += ray_counts[i * last_w + j];
                    // std::cout << ray_counts[i * last_w + j] << ' ';
                }
                // std::cout << std::endl;
            }
            delete[] ray_counts;
        }
        
        return ray_count;
    }

    void set_pos_target(const vec3 &pos, const vec3 &target_pos){
        position = pos;
        target_position = target_pos;
        build_lookat_mat();
    }

    vec3 cam_to_world(const vec3 &vec){
        return homogeneous_mult(lookat_mat, vec);
    }

    void render_gpu(const RenderOptions &opts, vec3 *frame_buffer, Scene &scene){
        ray_count = 0;
        last_w = opts.w;
        last_h = opts.h;

        if(!scene.gpu_ready()){
            scene.prepare_gpu();
        }

        if(dev_frame_buffer == nullptr){
            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_frame_buffer, opts.w * opts.h * sizeof(vec3)));
            CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_ray_counts, opts.w * opts.h * sizeof(vec3)));
        }

        float aspect_ratio = (float)opts.w / opts.h;
        float tan_fov = tan(opts.fov / 180.0f * M_PI / 2);

        KernelOptions kopts;
        kopts.frame_buffer = dev_frame_buffer;
        kopts.w = opts.w;
        kopts.h = opts.h;
        kopts.max_depth = opts.max_depth;
        kopts.background_color = opts.background_color;
        kopts.aspect_ratio = aspect_ratio;
        kopts.tan_fov = tan_fov;
        kopts.lookat_mat = lookat_mat;
        kopts.n_objects = scene.n_objects;
        kopts.n_lights = scene.n_lights;
        kopts.objects = scene.dev_objects;
        kopts.lights = scene.dev_lights;
        kopts.ray_counts = dev_ray_counts;

        render_kernel<<<dim3(32, 32), dim3(16, 16)>>>(kopts);
        // render_kernel<<<dim3(24, 24), dim3(24, 24)>>>(kopts);
        // render_kernel<<<dim3(16, 16), dim3(16, 16)>>>(kopts);
        CHECK_CUDA_CALL_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_KERNEL_ERROR();

        CHECK_CUDA_CALL_ERROR(cudaMemcpy(frame_buffer, dev_frame_buffer, opts.w * opts.h * sizeof(vec3), cudaMemcpyDeviceToHost));
    }

    void render(const RenderOptions &opts, vec3 *frame_buffer, const Scene &scene){
        ray_count = 0;
        float aspect_ratio = (float)opts.w / opts.h;
        float tan_fov = tan(opts.fov / 180.0f * M_PI / 2);

        for(int i = 0 ; i < opts.h; ++i){
            float pixel_y = (1.0f - 2.0f * ((i + 0.5f) / opts.h)) * tan_fov / aspect_ratio;
            for(int j = 0; j < opts.w; ++j){
                float pixel_x = (2.0f * ((j + 0.5f) / opts.w) - 1.0f) * tan_fov;
                vec3 pixel_world_pos = cam_to_world((vec3(pixel_x, pixel_y, -1.0)));
                vec3 camera_world_pos = cam_to_world(vec3(0.0f));

                vec3 ray_dir = normalize(pixel_world_pos - camera_world_pos);
                frame_buffer[i * opts.w + j] = cast_ray_host(camera_world_pos, ray_dir, scene.objects, scene.lights, 0, opts.max_depth, air_refractive_index, opts.background_color, &ray_count);
                // std::cout << "ray_count: " << ray_count << std::endl;
            }
        }
    }
};

#endif