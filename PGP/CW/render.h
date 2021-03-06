#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <limits>

#include "object.h"
#include "vec.h"
#include "mat.h"
#include "scene.h"
#include "utils.h"

#include "constants.h"

#define MAX_DEPTH 20

// host rendering functions
void triangle_intersection_host(const vec3 &origin, const vec3 &dir, const Object &obj, const Triangle &trig, float *t, float *u, float *v){
    vec3 e1 = obj.vertices[trig.v2] - obj.vertices[trig.v1];
    vec3 e2 = obj.vertices[trig.v3] - obj.vertices[trig.v1];

    mat3 m(-dir.x, e1.x, e2.x,
                -dir.y, e1.y, e2.y,
                -dir.z, e1.z, e2.z);
    
    vec3 temp = inv(m) * (origin - obj.vertices[trig.v1]);

    *t = temp.x;
    *u = temp.y;
    *v = temp.z;
}

bool shadow_ray_hit_host(const vec3 &origin, const vec3 &dir, const std::vector<Object> &objects, float *hit_t) {
    float t_min = std::numeric_limits<float>::max();
    bool hit = false;
    for(size_t obj_idx = 0; obj_idx < objects.size(); ++obj_idx){
        const Object &obj = objects[obj_idx];
        for(size_t trig_idx = 0; trig_idx < obj.trigs.size(); ++trig_idx){
            const Triangle &trig = obj.trigs[trig_idx];

            float t, u, v;
            triangle_intersection_host(origin, dir, obj, trig, &t, &u, &v);

            if(u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f){
                if(t < t_min){
                    t_min = t;
                }
                hit = true;
            }
        }
    }
    *hit_t = t_min;
    return hit;
}

vec3 phong_model_host(const vec3 &pos, const vec3 &direction, const Triangle &trig, float u, float v, 
    const Object &obj, const std::vector<Object> &objs, const std::vector<Light> &lights)
{
    vec3 normal = normalize((1.0f - u - v) * obj.normals[trig.n1] + u * obj.normals[trig.n2] + v * obj.normals[trig.n3]);

    vec3 ambient(obj.ka);
    vec3 diffuse(0.0f);
    vec3 specular(0.0f);

    for(size_t i = 0; i < lights.size(); ++i){
        vec3 light_pos = lights[i].pos;
        float hit_t = 0.0;
        vec3 L = light_pos - pos;
        float d = len(L);
        L = normalize(L); 

        if(shadow_ray_hit_host(light_pos, -L, objs, &hit_t) && (hit_t > d || (hit_t > d || (d - hit_t < eps)) ) ){
            float coef = lights[i].intensity / (d + K);
            diffuse += std::max(obj.kd * coef * dot_product(L, normal), 0.0f) * lights[i].color;
            vec3 R = normalize(reflect(-L, normal));
            vec3 S = -direction;
            specular += obj.ks * coef * std::pow(std::max(dot_product(R, S), 0.0f), obj.shininess) * lights[i].color; 
        }
    }

    if(obj.has_texture()){
        vec3 tex_pos = (1.0f - u - v) * obj.texture_coords[trig.t1] + u * obj.texture_coords[trig.t2] + v * obj.texture_coords[trig.t3];
        uchar4 color = obj.texture->get_color(tex_pos);
        return vec3((float)color.x / 255, (float)color.y / 255, (float)color.z / 255) * (diffuse + 0.2f * ambient) + obj.specular_color * specular;
    }

    return ambient * obj.ambient_color + diffuse * obj.diffuse_color + specular * obj.specular_color;
}

bool hit_host(const vec3 &origin, const vec3 &dir, const std::vector<Object> &objects,
    int *hit_obj_idx, vec3 *hit_pos, int *hit_triangle_idx, float *hit_u, float *hit_v, float *hit_t)
{
    float t_min = std::numeric_limits<float>::max();

    bool hit = false;
    for(size_t obj_idx = 0; obj_idx < objects.size(); ++obj_idx){
        const Object &obj = objects[obj_idx];
        for(size_t trig_idx = 0; trig_idx < obj.trigs.size(); ++trig_idx){
            const Triangle &trig = obj.trigs[trig_idx];

            float t, u, v;
            triangle_intersection_host(origin, dir, obj, trig, &t, &u, &v);

            if(u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f){
                if(t < t_min){
                    t_min = t;

                    *hit_obj_idx = obj_idx;
                    *hit_triangle_idx = trig_idx;
                    *hit_pos = origin + t * dir;
                    *hit_u = u;
                    *hit_v = v;
                }
                hit = true;
            }
        }
    }

    *hit_t = t_min;

    return hit;
}

vec3 cast_ray_host(const vec3 &origin, const vec3 &direction, const std::vector<Object> &objs, const std::vector<Light> &lights, 
    int cur_depth, int max_depth, float refractive_index, const vec3 &background_color, int *ray_count)
{
    (*ray_count)++;
    if(cur_depth >= max_depth)
        return background_color;

    int hit_obj_idx, hit_triangle_idx;
    float hit_u, hit_v, hit_t;
    vec3 hit_pos;
    if(hit_host(origin, direction, objs, &hit_obj_idx, &hit_pos, &hit_triangle_idx, &hit_u, &hit_v, &hit_t)){
        const Object &obj = objs[hit_obj_idx];
        const Triangle &trig = obj.trigs[hit_triangle_idx];

            vec3 normal = normalize((1.0f - hit_u - hit_v) * obj.normals[trig.n1] + hit_u * obj.normals[trig.n2] + hit_v * obj.normals[trig.n3]);

            vec3 main_color = phong_model_host(hit_pos, direction, trig, hit_u, hit_v, obj, objs, lights);
            float n1 = air_refractive_index, n2 = obj.refractive_index;
            float dp = dot_product(normal, direction);
            if(dp > 0){ // если луч выходит из среды в воздух
                std::swap(n1, n2);
            }

            vec3 refracted_direction, reflected_direction;
            if(dp > 0)
            {
                refracted_direction = refract(direction, -normal, n1, n2);
                reflected_direction = reflect(direction, -normal);
            }
            else{
                refracted_direction = refract(direction, normal, n1, n2);
                reflected_direction = reflect(direction, normal);
            }

            vec3 reflected_color = vec3(0.0f), refracted_color = vec3(0.0f);
            if(!approx_equal(obj.kt, 0.0f)){
                vec3 refracted_origin = dp < 0 ? hit_pos - eps * normal : hit_pos + eps * normal;
                refracted_color = cast_ray_host(refracted_origin, refracted_direction, objs, lights, cur_depth + 1, max_depth, n2, background_color, ray_count);
            }

            if(!approx_equal(obj.kr, 0.0f)){
                vec3 reflected_origin = dp < 0 ? hit_pos + eps * normal : hit_pos - eps * normal;
                reflected_color = cast_ray_host(reflected_origin, reflected_direction, objs, lights, cur_depth + 1, max_depth, refractive_index, background_color, ray_count);
            }

            return (1.0f - obj.kr - obj.kt) * main_color + obj.kr * reflected_color + obj.kt * obj.ambient_color * refracted_color;
    }

    return background_color;
}

// gpu rendering functions
struct KernelOptions{
    vec3 *frame_buffer;
    int w, h, max_depth;
    float aspect_ratio, tan_fov;
    vec3 background_color;
    mat4 lookat_mat;
    int n_objects;
    int n_lights;
    DevObject *objects;
    DevLight *lights;
    int *ray_counts;
};

__device__ void triangle_intersection_gpu(const vec3 &origin, const vec3 &dir, const DevObject &obj, const Triangle &trig, float *t, float *u, float *v){
    vec3 e1 = obj.vertices[trig.v2] - obj.vertices[trig.v1];
    vec3 e2 = obj.vertices[trig.v3] - obj.vertices[trig.v1];

    mat3 m(-dir.x, e1.x, e2.x,
                -dir.y, e1.y, e2.y,
                -dir.z, e1.z, e2.z);
    
    vec3 temp = inv(m) * (origin - obj.vertices[trig.v1]);

    *t = temp.x;
    *u = temp.y;
    *v = temp.z;
}

__device__ bool shadow_ray_hit_gpu(const vec3 &origin, const vec3 &dir, DevObject *objects, int n_objects, float *hit_t) {
    float t_min = FLT_MAX;

    bool hit = false;
    for(int obj_idx = 0; obj_idx < n_objects; ++obj_idx){
        const DevObject &obj = objects[obj_idx];
        for(int trig_idx = 0; trig_idx < obj.n_trigs; ++trig_idx){
            const Triangle &trig = obj.trigs[trig_idx];

            float t, u, v;
            triangle_intersection_gpu(origin, dir, obj, trig, &t, &u, &v);

            if(u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f){
                if(t < t_min){
                    t_min = t;
                }
                hit = true;
            }
        }
    }

    *hit_t = t_min;
    return hit;
}

__device__ bool hit_gpu(const vec3 &origin, const vec3 &dir, DevObject *objects, int n_objects,
    int *hit_obj_idx, vec3 *hit_pos, int *hit_triangle_idx, float *hit_u, float *hit_v, float *hit_t)
{
    float t_min = FLT_MAX;

    bool hit = false;
    for(int obj_idx = 0; obj_idx < n_objects; ++obj_idx){
        const DevObject &obj = objects[obj_idx];
        for(int trig_idx = 0; trig_idx < obj.n_trigs; ++trig_idx){
            const Triangle &trig = obj.trigs[trig_idx];

            float t, u, v;
            triangle_intersection_gpu(origin, dir, obj, trig, &t, &u, &v);

            if(u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f){
                if(t < t_min){
                    t_min = t;

                    *hit_obj_idx = obj_idx;
                    *hit_triangle_idx = trig_idx;
                    *hit_pos = origin + t * dir;
                    *hit_v = v;
                    *hit_u = u;
                }
                hit = true;
            }
        }
    }

    *hit_t = t_min;
    return hit;
}

struct context{
    vec3 origin;
    vec3 direction;
    vec3 color;
    int stage;
    int hit_obj_idx;
    int hit_trig_idx;
    float hit_t, hit_u, hit_v;
    vec3 coef;

    vec3 hit_pos;
    vec3 normal;
    float n1, n2;
};

__device__ vec3 phong_model_gpu(const vec3 &pos, const vec3 &direction, const Triangle &trig, float u, float v, 
    const DevObject &obj, DevObject *objs, int n_objects, DevLight *lights, int n_lights)
{
    vec3 normal = normalize((1.0f - u - v) * obj.normals[trig.n1] + u * obj.normals[trig.n2] + v * obj.normals[trig.n3]);

    vec3 ambient(obj.ka);
    vec3 diffuse(0.0f);
    vec3 specular(0.0f);

    for(int i = 0; i < n_lights; ++i){
        vec3 light_pos = lights[i].pos;
        float hit_t = 0.0;
        vec3 L = light_pos - pos;
        float d = len(L);
        L = normalize(L); 

        if(shadow_ray_hit_gpu(light_pos, -L, objs, n_objects, &hit_t) && (hit_t > d || (d - hit_t < eps)) ){
            float coef = lights[i].intensity / (d + K);
            diffuse += max(obj.kd * coef * dot_product(L, normal), 0.0f) * lights[i].color;
            vec3 R = normalize(reflect(-L, normal));
            vec3 S = -direction;
            specular += obj.ks * coef * pow(max(dot_product(R, S), 0.0f), obj.shininess) * lights[i].color; 
        }
    }

    if(obj.has_texture()){
        vec3 tex_pos = (1.0f - u - v) * obj.texture_coords[trig.t1] + u * obj.texture_coords[trig.t2] + v * obj.texture_coords[trig.t3];
        uchar4 color = obj.texture->get_color(tex_pos);
        return vec3((float)color.x / 255, (float)color.y / 255, (float)color.z / 255) * (diffuse + 0.2f * ambient) + obj.specular_color * specular;
    }
    return ambient * obj.ambient_color + diffuse * obj.diffuse_color + specular * obj.specular_color;
}

__device__ vec3 cast_ray_gpu(const vec3 &origin, vec3 &direction, const KernelOptions &opts, int *ray_count){
    
    int stack_top = 0;
    context stack[MAX_DEPTH];
    stack[0].origin = origin;
    stack[0].direction = direction;
    stack[0].color = vec3(0.0f);
    stack[0].stage = 0;

    int rc = 0;

    while(stack[0].stage < 3){
        context *top = &stack[stack_top];
        if(stack_top >= opts.max_depth){
            // "return" background color
            stack[stack_top - 1].color += top->coef * opts.background_color;
            stack[stack_top - 1].stage++;
            stack_top--;
        }
        else if(top->stage == 3){
            // return
            stack[stack_top - 1].color += top->coef * top->color;
            stack[stack_top - 1].stage++;
            stack_top--;
        }
        else if(top->stage == 0){
            int hit_obj_idx, hit_trig_idx;
            float hit_u, hit_v, hit_t;
            vec3 hit_pos;
            if(hit_gpu(top->origin, top->direction, opts.objects, opts.n_objects, &hit_obj_idx, &hit_pos, &hit_trig_idx, &hit_u, &hit_v, &hit_t)){

                top->hit_obj_idx = hit_obj_idx;
                top->hit_trig_idx = hit_trig_idx;
                top->hit_pos = hit_pos;
                top->hit_t = hit_t;
                top->hit_u = hit_u;
                top->hit_v = hit_v;

                DevObject &obj = opts.objects[hit_obj_idx];
                Triangle &trig = obj.trigs[hit_trig_idx];

                top->color = (1.0f - obj.kr - obj.kt) * phong_model_gpu(hit_pos, top->direction, trig, hit_u, hit_v, obj, opts.objects, opts.n_objects, opts.lights, opts.n_lights);
                
                vec3 normal = normalize((1.0f - hit_u - hit_v) * obj.normals[trig.n1] + hit_u * obj.normals[trig.n2] + hit_v * obj.normals[trig.n3]);
                
                top->normal = normal;
                
                float n1 = air_refractive_index, n2 = obj.refractive_index;
                float dp = dot_product(normal, top->direction);
                if(dp > 0){ // если луч выходит из среды в воздух
                    swap(&n1, &n2);
                }

                top->n1 = n1;
                top->n2 = n2;
                
                top->stage++;
            }
            else{
                // "return" background color
                top->color = opts.background_color;
                top->stage = 3;
            }

        }
        else if(top->stage == 1){
            // place new reflection task
            float dp = dot_product(top->normal, top->direction);
            vec3 reflected_origin, reflected_direction;
            if(dp > 0) {
                reflected_origin = top->hit_pos - eps * top->normal;
                reflected_direction = reflect(top->direction, -top->normal);
            }
            else{
                reflected_origin = top->hit_pos + eps * top->normal;
                reflected_direction = reflect(top->direction, top->normal);
            }

            if(!approx_equal(opts.objects[top->hit_obj_idx].kr, 0.0f)){
                stack_top++;
                stack[stack_top].stage = 0;
                stack[stack_top].coef = vec3(opts.objects[top->hit_obj_idx].kr);
                stack[stack_top].origin = reflected_origin;
                stack[stack_top].direction = reflected_direction;
                stack[stack_top].color = vec3(0.0f);
                rc++;
            }
            else{
                top->stage++;
            }
        }
        else if(top->stage == 2){
            // place new refraction task
            float dp = dot_product(top->normal, top->direction);

            vec3 refracted_origin, refracted_direction;
            if(dp > 0) {
                refracted_origin = top->hit_pos + eps * top->normal;
                refracted_direction = refract(top->direction, -top->normal, top->n1, top->n2);
            }
            else{
                refracted_origin = top->hit_pos - eps * top->normal;
                refracted_direction = refract(top->direction, top->normal, top->n1, top->n2);
            }

            if(!approx_equal(opts.objects[top->hit_obj_idx].kt, 0.0f)){
                stack_top++;
                stack[stack_top].stage = 0;
                stack[stack_top].coef = vec3(opts.objects[top->hit_obj_idx].kt) * opts.objects[top->hit_obj_idx].ambient_color;
                stack[stack_top].origin = refracted_origin;
                stack[stack_top].direction = refracted_direction;
                stack[stack_top].color = vec3(0.0f);
                rc++;
            }
            else{
                top->stage++;
            }
        }
    }

    // printf("rc: %d\n", rc);

    *ray_count = rc;
    return stack[0].color;
}

__global__ void render_kernel(KernelOptions opts){
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int i = id_y ; i < opts.h; i += offset_y){
        float pixel_y = (1.0f - 2.0f * ((i + 0.5f) / opts.h)) * opts.tan_fov / opts.aspect_ratio;
        for(int j = id_x; j < opts.w; j += offset_x){
            float pixel_x = (2.0f * ((j + 0.5f) / opts.w) - 1.0f) * opts.tan_fov;
            
            vec3 pixel_world_pos = homogeneous_mult(opts.lookat_mat, vec3(pixel_x, pixel_y, -1.0));
            // vec3 pixel_world_pos = homogeneous_mult(opts.lookat_mat, vec3(pixel_x, pixel_y, -1.0));
            vec3 camera_world_pos = homogeneous_mult(opts.lookat_mat, vec3(0.0f));
            
            int ray_count;
            
            vec3 ray_dir = normalize(pixel_world_pos - camera_world_pos);
            opts.frame_buffer[i * opts.w + j] = cast_ray_gpu(camera_world_pos, ray_dir, opts, &ray_count);
            // opts.frame_buffer[i * opts.w + j] = cast_ray_gpu(camera_world_pos, ray_dir, opts, &opts.ray_counts[i * opts.w + j]);
            
            // ray_count++;
            // __syncthreads();
            // printf("ray_count: %d\n", ray_count);
            opts.ray_counts[i * opts.w + j] = ray_count + 1;
        }
    }
}

#endif