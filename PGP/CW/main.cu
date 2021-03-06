#include <stdio.h>
#include <vector>
#include <map>
#include <float.h>
#include <tuple>
#include <cstring>
#include <chrono>
#include <linux/limits.h>

#include "vec.h"
#include "mat.h"
#include "object.h"
#include "error.h"
#include "constants.h"
#include "camera.h"
#include "scene.h"
#include "image.h"
#include "ssaa.h"

Object disc(float r, int n_segments = 8){
    Object obj;
    float d_angle = 2.0 * M_PI / n_segments;
    obj.vertices.push_back(vec3(0.0f));
    obj.texture_coords.push_back(vec3(0.5f, 0.5f, 0.0f));

    obj.normals.push_back(vec3(0.0f, 1.0f, 0.0f));
    for(int i = 0; i < n_segments; ++i){
        float c = cos(d_angle * i), s = sin(d_angle * i);

        obj.vertices.push_back(vec3(r * c, 0.0f, r * s));
        obj.texture_coords.push_back(vec3(0.5f + 0.5f * c, 0.5f + 0.5f * s, 0.0f));
        if(i > 0){
            obj.trigs.push_back(Triangle(i+1, i, 0, 0, 0, 0, i+1, i, 0));
        }
    }
    obj.trigs.push_back(Triangle(1, n_segments, 0, 0, 0, 0, 1, n_segments, 0));

    return obj;
}

Object quadrangle(const vec3 &v1, const vec3 &v2, const vec3 &v3, const vec3 &v4){
    vec3 e2 = v2 - v1, e3 = v3 - v1, e4 = v4 - v1;
    vec3 normal1 = normalize(cross_product(e3, e2));
    vec3 normal2 = normalize(cross_product(e4, e3));

    Object res;
    res.vertices = {v1, v2, v3, v4};
    res.normals = {normal1, normal2};
    res.texture_coords = {vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(1.0f, 1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)};
    res.trigs = {Triangle(0, 2, 1, 0, 0, 0, 0, 2, 1), Triangle(0, 3, 2, 1, 1, 1, 0, 3, 2)};

    return res;
}

std::vector<Object> gen_lights(const Object &frame, float r, float a, float margin, float offset, int n_lights, float light_r){
    std::vector<vec3> end_points;

    vec3 center(0.0f);
    int n_vertices = frame.vertices.size();
    for(int i = 0; i < n_vertices; ++i)
        center += frame.vertices[i];

    center = (1.0f / n_vertices) * center;
    float eps = 0.01;

    for(int i = 0; i < n_vertices; ++i){
        if(approx_equal(len(center - frame.vertices[i]), r, eps)){
            end_points.push_back(frame.vertices[i]);
        }
    }
    
    std::vector<std::pair<size_t, size_t>> edges;
    for(size_t i = 0; i < end_points.size(); ++i){
        for(size_t j = 0; j < end_points.size(); ++j){
            if(approx_equal(len(end_points[i] - end_points[j]), a, eps)){
                bool used = false;
                for(size_t k = 0; k < edges.size(); ++k){
                    if(edges[k].first == j && edges[k].second == i){
                        used = true;
                        break;
                    }
                }
                if(!used){
                    edges.push_back(std::make_pair(i, j));
                }
            }
        }
    }

    float line_len = a - 2 * margin;
    float d = line_len / (n_lights - 1);

    Object dsc = disc(light_r, 8);
    std::vector<Object> res;

    for(size_t i = 0; i < edges.size(); ++i){
        int n1 = edges[i].first, n2 = edges[i].second;
        vec3 e1 = end_points[n1], e2 = end_points[n2];

        vec3 normal = normalize(0.5f * (center - e1 + (center - e2)));

        vec3 dir = e2 - e1;
        vec3 dir_normalized = normalize(dir);
        vec3 start_point = e1 + dir_normalized * margin;
        for(int j = 0; j < n_lights; ++j){
            vec3 pos = start_point + d * j * dir_normalized + normal * offset;

            Object tmp = dsc;

            // rotate to align normal vector of light with normal
            mat3 align = align_mat(vec3(0.0f, 1.0f, 0.0f), normal);
            tmp.rotate(align);
            tmp.translate(pos);

            // translate light to (pos position + normal * offset)
            res.push_back(tmp);
        }
    }

    return res;
}

vec3 cylindrical2decart(const vec3 &p){
    return {p.x * cos(p.y), p.x * sin(p.y), p.z};
}

void print_default(){
    std::cout << "300\n"
    "./PIC%03d.data\n"
    "640 480 90\n"
    "4.1 1.0 0.0\t2.0 0.3\t1.0 1.0 2.0\t0.0 0.0\n"
    "1.0 0.0 0.0\t0.5 0.1\t1.0 1.0 2.0\t0.0 0.0\n"
    "3.0 3.0 0.0\t1.0 0.0 0.0\t2.0\t0.1\t0.9\t10\n"
    "0.0 0.0 0.0\t0.0 1.0 0.0\t1.5\t0.2\t0.8\t5\n"
    "-3.0 -3.0 0.0\t0.0 0.7 0.7\t1.0\t0.3\t0.7\t4\n"
    "-5.0 -5.0 -2.1\t-5.0 5.0 -2.1\t5.0 5.0 -2.1\t5.0 -5.0 -2.1\t./texture4.data\t1.0 1.0 1.0\t0.3\n"
    "4\n"
    "-5.0 -5.0 5.0\t1.0 1.0 1.0\n"
    "5.0 5.0 5.0\t1.0 1.0 1.0\n"
    "-5.0 5.0 5.0\t1.0 0.996 0.890\n"
    "5.0 -5.0 5.0\t1.0 0.996 0.890\n"
    "7 1" << std::endl;
}

int main(int argc, char **argv){
    if(argc < 2 || argc > 3){
        std::cout << "Usage: " << argv[0] << " (--gpu | --cpu | --default) [--ssaa]" << std::endl;
        return 0;
    }

    bool use_gpu = true, use_ssaa = false;
    for(int i = 1; i < argc; ++i){
        if(!strcmp(argv[i], "--gpu")){
            use_gpu = true;
        }
        else if(!strcmp(argv[i], "--cpu")){
            use_gpu = false;
        }
        else if(!strcmp(argv[i], "--ssaa")){
            use_ssaa = true;
        }
        else if(!strcmp(argv[i], "--default")){
            print_default();
            return 0;
        }
    }

    int n_frames;
    std::string path;
    int w, h;
    float fov;
    float r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic;
    float prc, pzc, r0n, z0n, phi0n;
    float Arn, Azn, wrn, wzn, wphin, prn, pzn;

    vec3 tetra_pos, tetra_color; 
    vec3 cube_pos, cube_color; 
    vec3 octa_pos, octa_color; 

    float tetra_r, cube_r, octa_r;
    float tetra_kr, cube_kr, octa_kr;
    float tetra_kt, cube_kt, octa_kt;
    int tetra_n_lights, cube_n_lights, octa_n_lights; 

    vec3 floor_v1, floor_v2, floor_v3, floor_v4;
    vec3 floor_color;
    std::string floor_texture_path;
    float floor_kr;

    std::cin >> n_frames;
    std::cin >> path;
    std::cin >> w >> h >> fov;
    
    std::cout << "n_frames: " << n_frames << std::endl;
    std::cout << "path: " << path << std::endl;
    std::cout << "w: " << w << " h: " << h << " fov: " << fov << std::endl;

    std::cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc;
    std::cout << "r0c: " << r0c << " z0c: " << z0c << " phi0c: " << phi0c << std::endl;
    std::cout << "Arc: " << Arc << " Azc: " << Azc << std::endl;
    std::cout << "wrc: " << wrc << " wzc: " << wzc << " wphic: " << wphic << std::endl;
    std::cout << "prc: " << prc << " pzc: " << pzc << std::endl;

    std::cin >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;
    std::cout << "r0n: " << r0n << " z0n: " << z0n << " phi0n: " << phi0n << std::endl;
    std::cout << "Arn: " << Arn << " Azn: " << Azn << std::endl;
    std::cout << "wrn: " << wrn << " wzn: " << wzn << " wphin: " << wphin << std::endl;
    std::cout << "prn: " << prn << " pzn: " << pzn << std::endl;

    std::cin >> tetra_pos >> tetra_color >> tetra_r >> tetra_kr >> tetra_kt >> tetra_n_lights;
    std::cin >> cube_pos >> cube_color >> cube_r >> cube_kr >> cube_kt >> cube_n_lights;
    std::cin >> octa_pos >> octa_color >> octa_r >> octa_kr >> octa_kt >> octa_n_lights;
    
    std::cin >> floor_v1 >> floor_v2 >> floor_v3 >> floor_v4 >> floor_texture_path >> floor_color >> floor_kr;
    
    std::cout << "floor: " << std::endl;
    std::cout << "v1: " << floor_v1 << std::endl;
    std::cout << "v2: " << floor_v2 << std::endl;
    std::cout << "v3: " << floor_v3 << std::endl;
    std::cout << "v4: " << floor_v4 << std::endl;

    int n_lights;
    std::cin >> n_lights;
    std::cout << "n_lights: " << n_lights << std::endl;

    Scene scene;

    std::cout << "lights: " << std::endl;
    float light_intensity = 10.0f;
    for(int i = 0; i < n_lights; ++i){
        vec3 pos, color;
        std::cin >> pos >> color;
        std::cout << "position: " << pos << " color: " << color << " intensity: " << light_intensity << std::endl;
        scene.add_light(Light(pos, color, light_intensity));
    }

    int max_depth, aa_kernel_size;
    std::cin >> max_depth >> aa_kernel_size;
    std::cout << "max_depth: " << max_depth << std::endl;
    std::cout << "anti-aliasing kernel size: " << aa_kernel_size << std::endl;
    
    vec3 frame_ambient_color = vec3(0.1f);
    vec3 frame_diffuse_color = vec3(0.1f);
    vec3 frame_specular_color = vec3(1.0f);

    float frame_ka = 0.2f;
    float frame_kd = 0.4f;
    float frame_ks = 1.0f;

    float frame_kr = 0.0f;
    float frame_kt = 0.0f;
    float frame_refractive_index = 1.0f;
    float frame_shininess = 32.0f;

    vec3 glasses_specular_color = vec3(1.0f);
    float glasses_ka = 1.0f;
    float glasses_kd = 0.2;
    float glasses_ks = 1.0f;
    float glasses_refractive_index = 1.5f;

    Object octa_frame = import_obj("octa_frame.obj").scale(vec3(octa_r)).translate(octa_pos);
    octa_frame.ambient_color = frame_ambient_color;
    octa_frame.diffuse_color = frame_diffuse_color;
    octa_frame.specular_color = frame_specular_color;
    
    octa_frame.ka = frame_ka;
    octa_frame.kd = frame_kd;
    octa_frame.ks = frame_ks;
    octa_frame.shininess = frame_shininess;

    octa_frame.kr = frame_kr;
    octa_frame.kt = frame_kt;
    octa_frame.refractive_index = frame_refractive_index;

    Object octa_glasses = import_obj("octa_glasses.obj").scale(vec3(octa_r)).translate(octa_pos);
    octa_glasses.ambient_color = octa_color;
    octa_glasses.diffuse_color = octa_color;
    octa_glasses.specular_color = glasses_specular_color;
    
    octa_glasses.kr = octa_kr;
    octa_glasses.kt = octa_kt;
    octa_glasses.refractive_index = glasses_refractive_index;

    octa_glasses.ka = glasses_ka;
    octa_glasses.kd = glasses_kd;
    octa_glasses.ks = glasses_ks;
    
    Object cube_frame = import_obj("cube_frame.obj").scale(vec3(cube_r)).translate(cube_pos);
    cube_frame.ambient_color = frame_ambient_color;
    cube_frame.diffuse_color = frame_diffuse_color;
    cube_frame.specular_color = frame_specular_color;
    
    cube_frame.ka = frame_ka;
    cube_frame.kd = frame_kd;
    cube_frame.ks = frame_ks;
    cube_frame.shininess = frame_shininess;

    cube_frame.kr = frame_kr;
    cube_frame.kt = frame_kt;
    cube_frame.refractive_index = frame_refractive_index;

    Object cube_glasses = import_obj("cube_glasses.obj").scale(vec3(cube_r)).translate(cube_pos);
    cube_glasses.ambient_color = cube_color;
    cube_glasses.diffuse_color = cube_color;
    cube_glasses.specular_color = glasses_specular_color;
    
    cube_glasses.kr = cube_kr;
    cube_glasses.kt = cube_kt;
    cube_glasses.refractive_index = glasses_refractive_index;

    cube_glasses.ka = glasses_ka;
    cube_glasses.kd = glasses_kd;
    cube_glasses.ks = glasses_ks;

    Object tetra_frame = import_obj("tetra_frame.obj").scale(vec3(tetra_r)).rotate(vec3(M_PI / 2.0f, 0.0f, 0.0f)).translate(tetra_pos);
    tetra_frame.ambient_color = frame_ambient_color;
    tetra_frame.diffuse_color = frame_diffuse_color;
    tetra_frame.specular_color = frame_specular_color;
    
    tetra_frame.kr = frame_kr;
    tetra_frame.kt = frame_kt;
    tetra_frame.refractive_index = frame_refractive_index;

    tetra_frame.ka = frame_ka;
    tetra_frame.kd = frame_kd;
    tetra_frame.ks = frame_ks;

    Object tetra_glasses = import_obj("tetra_glasses.obj").scale(vec3(tetra_r)).rotate(vec3(M_PI / 2.0f, 0.0f, 0.0f)).translate(tetra_pos);
    tetra_glasses.ambient_color = tetra_color;
    tetra_glasses.diffuse_color = tetra_color;
    tetra_glasses.specular_color = glasses_specular_color;
    
    tetra_glasses.kr = tetra_kr;
    tetra_glasses.kt = tetra_kt;
    tetra_glasses.refractive_index = glasses_refractive_index;

    tetra_glasses.ka = glasses_ka;
    tetra_glasses.kd = glasses_kd;
    tetra_glasses.ks = glasses_ks;
    
    Object floor = quadrangle(floor_v1, floor_v2, floor_v3, floor_v4);
    
    Texture floor_texture = Texture::import_data(floor_texture_path);
    floor_texture.change_color(floor_color);

    floor.assign_texture(&floor_texture);
    floor.ambient_color = vec3(0.1f, 0.0f, 0.0f);
    floor.diffuse_color = vec3(1.0f, 0.0f, 0.0f);
    floor.specular_color = vec3(1.0f);

    floor.ka = 0.2f;
    floor.kd = 0.4f;
    floor.ks = 0.8f;
    floor.shininess = 128.0f;

    floor.kr = floor_kr;
    floor.kt = 0.0f;
    floor.refractive_index = 1.0f;

    float lights_ambient = 10.0f;
    float lights_diffuse = 10.0f;
    float lights_specular = 10.0f;

    float octa_a = 2.0f / sqrt(2.0f) * octa_r;
    std::vector<Object> octa_lights = gen_lights(octa_frame, octa_r, octa_a, 0.2f * octa_r, 0.06f * octa_r, octa_n_lights, 0.015f * octa_r);

    for(size_t i = 0; i < octa_lights.size(); ++i){
        octa_lights[i].kr = 0.0f;
        octa_lights[i].kt = 0.0f;
        octa_lights[i].refractive_index = 1.0f;

        octa_lights[i].ka = 1.0f;
        octa_lights[i].kd = 0.0f;
        octa_lights[i].ks = 0.0f;
        octa_lights[i].shininess = 2.0f;

        octa_lights[i].ambient_color = vec3(lights_ambient);
        octa_lights[i].diffuse_color = vec3(lights_diffuse);
        octa_lights[i].specular_color = vec3(lights_specular);

        scene.add_object(octa_lights[i]);
    }

    float cube_a = 2.0f / sqrt(3.0f) * cube_r;
    std::vector<Object> cube_lights = gen_lights(cube_frame, cube_r, cube_a, 0.2f * cube_r, 0.08f * cube_r, cube_n_lights, 0.015f * cube_r);

    for(size_t i = 0; i < cube_lights.size(); ++i){
        cube_lights[i].kr = 0.0f;
        cube_lights[i].kt = 0.0f;
        cube_lights[i].refractive_index = 1.0f;

        cube_lights[i].ka = 1.0f;
        cube_lights[i].kd = 0.0f;
        cube_lights[i].ks = 0.0f;
        cube_lights[i].shininess = 2.0f;

        cube_lights[i].ambient_color = vec3(lights_ambient);
        cube_lights[i].diffuse_color = vec3(lights_diffuse);
        cube_lights[i].specular_color = vec3(lights_specular);

        scene.add_object(cube_lights[i]);
    }

    float tetra_a = 4.0f / sqrt(6.0f) * tetra_r;
    std::vector<Object> tetra_lights = gen_lights(tetra_frame, tetra_r, tetra_a, 0.2f * tetra_r, 0.08f * tetra_r, tetra_n_lights, 0.015f * tetra_r);

    for(size_t i = 0; i < tetra_lights.size(); ++i){
        tetra_lights[i].kr = 0.0f;
        tetra_lights[i].kt = 0.0f;
        tetra_lights[i].refractive_index = 1.0f;

        tetra_lights[i].ka = 1.0f;
        tetra_lights[i].kd = 0.0f;
        tetra_lights[i].ks = 0.0f;
        tetra_lights[i].shininess = 2.0f;

        tetra_lights[i].ambient_color = vec3(lights_ambient);
        tetra_lights[i].diffuse_color = vec3(lights_diffuse);
        tetra_lights[i].specular_color = vec3(lights_specular);

        scene.add_object(tetra_lights[i]);
    }

    scene.add_object(octa_frame);
    scene.add_object(octa_glasses);

    scene.add_object(cube_frame);
    scene.add_object(cube_glasses);

    scene.add_object(tetra_frame);
    scene.add_object(tetra_glasses);
    
    scene.add_object(floor);

    int aa_w = w, aa_h = h;
    vec3 *frame_buffer = nullptr, *aa_frame_buffer = nullptr;

    if(use_ssaa){
        aa_w = w;
        aa_h = h;
        w *= aa_kernel_size;
        h *= aa_kernel_size;
        std::cout << "using super sampling anti-aliasing" << std::endl;
        std::cout << "frame_buffer: " << "w: " << w << " h: " << h << std::endl;
        aa_frame_buffer = new vec3[aa_w * aa_h];
    }
    frame_buffer = new vec3[w * h];

    RenderOptions opts;
    opts.background_color = vec3(0.0f);
    opts.fov = fov;
    opts.w = w;
    opts.h = h;
    opts.max_depth = max_depth;

    float t = 0.0;
    float dt = 2.0f * M_PI / n_frames;

    char *img_path = new char[PATH_MAX];

    vec3 camera_position = vec3(0.0), target_position = vec3(0.0);
    
    Camera cam(camera_position, target_position);
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    if(use_gpu){
        std::cout << "Rendering on GPU..." << std::endl;
    }
    else{
        std::cout << "Rendering on CPU..." << std::endl;
    }

    std::cout << "frame #\t\t" << "time(ms)\t" << "rays count" << std::endl;
    for(int i = 0; i < n_frames; ++i){
        camera_position = cylindrical2decart(vec3(r0c + Arc * sin(wrc * t + prc), phi0c + wphic * t, z0c + Azc * sin(wzc * t + pzc)));
        target_position = cylindrical2decart(vec3(r0n + Arn * sin(wrn * t + prn), phi0n + wphin * t, z0n + Azn * sin(wzn * t + pzn)));
        
        sprintf(img_path, path.c_str(), i);

        cam.set_pos_target(camera_position, target_position);

        start = std::chrono::high_resolution_clock::now();
        if(use_gpu){
            cam.render_gpu(opts, frame_buffer, scene);
            if(use_ssaa){
                SSAA_gpu(frame_buffer, aa_frame_buffer, w, h, aa_w, aa_h);
                save_data(aa_frame_buffer, aa_w, aa_h, img_path);
            }
            else{
                save_data(frame_buffer, w, h, img_path);
            }
        }
        else{
            cam.render(opts, frame_buffer, scene);
            if(use_ssaa){
                SSAA(frame_buffer, aa_frame_buffer, w, h, aa_w, aa_h);
                save_data(aa_frame_buffer, aa_w, aa_h, img_path);
            }
            else{
                save_data(frame_buffer, w, h, img_path);
            }
        }
        end = std::chrono::high_resolution_clock::now();

        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << i << "\t\t" << elapsed_time << "\t\t" << cam.get_ray_count() << std::endl;
        t += dt;
    }

    delete[] frame_buffer;
    delete[] img_path;
    if(aa_frame_buffer != nullptr)
        delete[] aa_frame_buffer;
}