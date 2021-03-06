#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "error.h"

#define N_PARTICLES 300
#define eps 0.001
#define K 50.0f

int prev_time;

int w = 1280;
int h = 720;

float fov = 60.0f;

float deceleration = 0.9;

float aspect_ratio = (float)w / h;

float x = -31.0f, y = 0.0f, z = 12.0f;
float dx = 0.0f, dy = 0.0f, dz = 0.0f;

float yaw = 0.0f, pitch = -0.42f;
float dyaw = 0.0f, dpitch = 0.0f;

float camera_speed = 20.0f;
float sensivity = 0.1f;

float g = 20.0f;

float bullet_speed = 50.0f;

float particle_q = 1.0f;
float camera_q = 30.0f;
float bullet_q = 50.0f;

int mouse_x_prev = 0;
int mouse_y_prev = 0;

float restoring_velocity = 10.f;

#define a 15.0f

GLUquadric* sphere;

#define floor_texture_size 200

__const__ float shift_z = 0.75f;

GLuint particle_texture;
GLuint floor_texture;
GLuint vbo;

cudaGraphicsResource *res;		

struct particle_t{
	float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    float vax, vay, vaz;
	float q;
};

particle_t particle[N_PARTICLES];

#define N_MAX_BULLETS 30

particle_t bullet[N_MAX_BULLETS];
int bullet_time[N_MAX_BULLETS];
int bullet_used_slot[N_MAX_BULLETS];

char title[50];

__constant__ particle_t dev_particles[N_PARTICLES + N_MAX_BULLETS];

__global__ void kernel(uchar4 *data, int n_active_particles) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	float x, y;
	for(i = idx; i < floor_texture_size; i += offsetx)
		for(j = idy; j < floor_texture_size; j += offsety) {
            x = 2 * a * (((float)i / floor_texture_size) - 0.5);
            y = 2 * a * (((float)j / floor_texture_size) - 0.5);

            float E = 0.0f;
            for(int k = 0; k < n_active_particles; ++k){
                float dx = dev_particles[k].x - x;
                float dy = dev_particles[k].y - y;
                float dz = dev_particles[k].z - shift_z;

                E += dev_particles[k].q / (dx*dx + dy*dy + dz*dz + eps);
            }
            data[j * floor_texture_size + i] = make_uchar4(0, 0, (unsigned char)min(E * K, 255.f), 255);
        }
}

float clamp(float val, float min_val, float max_val){
    return min(max(val, min_val), max_val);
}

float distance(float ax, float ay, float az, float bx, float by, float bz){
    float dx = ax - bx;
    float dy = ay - by;
    float dz = az - bz;

    return sqrt(dx*dx + dy*dy + dz*dz);
}

float random_range(float min_val, float max_val){
    return min_val + (float)rand() / RAND_MAX * (max_val - min_val);
}

void init(void) {
    prev_time = glutGet(GLUT_ELAPSED_TIME);

    srand(time(0));
    rand();
    for(int i = 0; i < N_PARTICLES; ++i){
        particle[i].x = random_range(-a + eps, a - eps);
        particle[i].y = random_range(-a + eps, a - eps);
        particle[i].z = random_range(eps, 2 * a - eps);

        particle[i].vx = 0.0f;
        particle[i].vy = 0.0f;
        particle[i].vz = 0.0f;

        particle[i].ax = random_range(0, 360.0f);
        particle[i].ay = random_range(0, 360.0f);
        particle[i].az = random_range(0, 360.0f);

        particle[i].vax = random_range(0, 360.0f);
        particle[i].vay = random_range(0, 360.0f);
        particle[i].vaz = random_range(0, 360.0f);

        particle[i].q = particle_q;
    }

    for(int i = 0; i < N_MAX_BULLETS; ++i){
        bullet_used_slot[i] = 0;
    }
}

int n_updates = 0;
int max_n_updates = 100;

float dt_accum = 0.0f;

void camera_direction(float *x, float *y, float *z){
    *x = cos(pitch) * cos(yaw);
    *y = cos(pitch) * sin(yaw);
    *z = sin(pitch);
}

void update(void) {
    n_updates++;
    int time;

    float dt;
    time = glutGet(GLUT_ELAPSED_TIME);
    dt = (time - prev_time) / 1000.0;
    prev_time = time;

    dt_accum += dt;

    if(n_updates == max_n_updates){
        sprintf(title, "Updates per second: %.2f", n_updates / dt_accum);
        glutSetWindowTitle(title);

        dt_accum = 0.0f;
        n_updates = 0;
    }

    x += dx * dt;
    y += dy * dt;
    z += dz * dt;

    yaw += dyaw * dt;
    pitch = clamp(pitch + dpitch * dt, -M_PI / 2 + eps, M_PI / 2 - eps);
    
    dx *= deceleration;
    dy *= deceleration;
    dz *= deceleration;

    dyaw = 0.0f;
    dpitch = 0.0f;

    for(int i = 0; i < N_MAX_BULLETS; ++i){
        if(bullet_used_slot[i]){
            if(time - bullet_time[i] > 1000){
                bullet_used_slot[i] = 0;
                break;
            }

            bullet[i].x += bullet[i].vx * dt;
            bullet[i].y += bullet[i].vy * dt;
            bullet[i].z += bullet[i].vz * dt;
        }
    }

    for(int i = 0; i < N_PARTICLES; ++i){
        float interaction_sum_x = 0.0f;
        float interaction_sum_y = 0.0f;
        float interaction_sum_z = 0.0f;


        // расчет воздействия частиц
        float lir = particle[i].x - a; // расстояние от i-й частицы до правой стенки
        float lil = particle[i].x + a; // расстояние от i-й частицы до левой стенки

        float lif = particle[i].y - a; // расстояние от i-й частицы до передней стенки
        float lib = particle[i].y + a; // расстояние от i-й частицы до задней стенки

        float lid = particle[i].z - 2 * a; // расстояние от i-й частицы до нижней стенки
        float liu = particle[i].z; // расстояние от i-й частицы до верхней стенки

        for(int j = 0; j < N_PARTICLES; ++j){
            float lij = distance(particle[i].x, particle[i].y, particle[i].z,
                particle[j].x, particle[j].y, particle[j].z);

            interaction_sum_x += particle[j].q * (particle[i].x - particle[j].x) / (lij*lij*lij + eps);
            interaction_sum_y += particle[j].q * (particle[i].y - particle[j].y) / (lij*lij*lij + eps);
            interaction_sum_z += particle[j].q * (particle[i].z - particle[j].z) / (lij*lij*lij + eps);
        }

        // расчет воздействия частиц-снарядов
        for(int j = 0; j < N_MAX_BULLETS; ++j){
            if(!bullet_used_slot[j])
                continue;

            float lij = distance(particle[i].x, particle[i].y, particle[i].z,
                bullet[j].x, bullet[j].y, bullet[j].z);

            interaction_sum_x += bullet[j].q * (particle[i].x - bullet[j].x) / (lij*lij*lij + eps);
            interaction_sum_y += bullet[j].q * (particle[i].y - bullet[j].y) / (lij*lij*lij + eps);
            interaction_sum_z += bullet[j].q * (particle[i].z - bullet[j].z) / (lij*lij*lij + eps);
        }

        // расчет воздействия камеры
        float lic = distance(x, y, z, particle[i].x, particle[i].y, particle[i].z);

        interaction_sum_x += camera_q * (particle[i].x - x) / (lic*lic*lic + eps);
        interaction_sum_y += camera_q * (particle[i].y - y) / (lic*lic*lic + eps);
        interaction_sum_z += camera_q * (particle[i].z - z) / (lic*lic*lic + eps);

        // расчет стенок куба
        interaction_sum_x += particle[i].q * lir / (abs(lir*lir*lir) + eps) + particle[i].q * lil / (abs(lil*lil*lil) + eps);
        interaction_sum_y += particle[i].q * lif / (abs(lif*lif*lif) + eps) + particle[i].q * lib / (abs(lib*lib*lib) + eps);
        interaction_sum_z += particle[i].q * liu / (abs(liu*liu*liu) + eps) + particle[i].q * lid / (abs(lid*lid*lid) + eps);

        float W = 0.99;

        particle[i].vx *= W;
        particle[i].vy *= W;
        particle[i].vz *= W;
        
        particle[i].vx += K * particle[i].q * interaction_sum_x * dt;
        particle[i].vy += K * particle[i].q * interaction_sum_y * dt;
        particle[i].vz += K * particle[i].q * interaction_sum_z * dt - g * dt;
                           
        // возвращающая внутрь куба "сила"
        if(particle[i].x < -a){
            particle[i].vx += restoring_velocity;
        }

        if(particle[i].x > a){
            particle[i].vx -= restoring_velocity;
        }

        if(particle[i].y < -a){
            particle[i].vy += restoring_velocity;
        }

        if(particle[i].y > a){
            particle[i].vy -= restoring_velocity;
        }

        if(particle[i].z < 0.0f){
            particle[i].vz += restoring_velocity;
        }

        if(particle[i].z > 2 * a){
            particle[i].vz -= restoring_velocity;
        }
                                                            
        particle[i].x += particle[i].vx * dt;
        particle[i].y += particle[i].vy * dt;
        particle[i].z += particle[i].vz * dt;

        particle[i].ax += particle[i].vax * dt;
        particle[i].ay += particle[i].vay * dt;
        particle[i].az += particle[i].vaz * dt;
    }

    CHECK_CUDA_CALL_ERROR(cudaMemcpyToSymbol(dev_particles, particle, N_PARTICLES * sizeof(particle_t), 0, cudaMemcpyHostToDevice));

    int n_active_bullets = 0;
    for(int i = 0; i < N_MAX_BULLETS; ++i){
        if(!bullet_used_slot[i])
            continue;
        
        CHECK_CUDA_CALL_ERROR(cudaMemcpyToSymbol(dev_particles, &bullet[i], sizeof(particle_t), N_PARTICLES * sizeof(particle_t), cudaMemcpyHostToDevice));
        n_active_bullets++;
    }
    
	uchar4* dev_data;
    size_t size;
    
	CHECK_CUDA_CALL_ERROR(cudaGraphicsMapResources(1, &res, 0));		// Делаем буфер доступным для CUDA
	CHECK_CUDA_CALL_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));	// Получаем указатель на память буфера
    kernel<<<dim3(32, 32), dim3(32, 8)>>>(dev_data, N_PARTICLES + n_active_bullets);
    CHECK_CUDA_KERNEL_ERROR();		
	CHECK_CUDA_CALL_ERROR(cudaGraphicsUnmapResources(1, &res, 0));		// Возращаем буфер OpenGL'ю что бы он мог его использоват
}

void shot(){
    int i = 0;
    for(; i < N_MAX_BULLETS; ++i)
        if(!bullet_used_slot[i])
            break;

    bullet_used_slot[i] = 1;

    if(i == N_MAX_BULLETS)
        return;

    float vx, vy, vz;
    camera_direction(&vx, &vy, &vz);

    bullet[i].x = x + vx * 0.01;
    bullet[i].y = y + vy * 0.01;
    bullet[i].z = z + vz * 0.01;

    bullet[i].vx = vx * bullet_speed;
    bullet[i].vy = vy * bullet_speed;
    bullet[i].vz = vz * bullet_speed;

    bullet[i].q = bullet_q;

    bullet_time[i] = glutGet(GLUT_ELAPSED_TIME);
}

void mouse_movement(int mouse_x, int mouse_y){
    dyaw -= (mouse_x - w / 2) * sensivity;
    dpitch -= (mouse_y - h / 2) * sensivity;

	if(mouse_x != w / 2 || mouse_y != h / 2) {
        glutWarpPointer(w / 2, h / 2);
    }
}

void mouse_clicks(int button, int state, int mouse_x, int mouse_y){
	if(button == GLUT_LEFT_BUTTON && state ==  GLUT_DOWN) {
        shot();
    }
}

void keys(unsigned char key, int x, int y) {
	if(key == 27) {
		CHECK_CUDA_CALL_ERROR(cudaGraphicsUnregisterResource(res));
		glBindBuffer(1, vbo);
        glDeleteBuffers(1, &vbo);
        gluDeleteQuadric(sphere);
        glDeleteTextures(1, &particle_texture);
		glDeleteTextures(1, &floor_texture);
		exit(0);
    }
    else if(key == 'w') {
        dx += cos(pitch) * cos(yaw) * camera_speed;
        dy += cos(pitch) * sin(yaw) * camera_speed;
        dz += sin(pitch) * camera_speed;
    }
    else if(key == 'a') {
        dx += -sin(yaw) * camera_speed;
        dy += cos(yaw) * camera_speed;
    }
    else if(key == 's') {
        dx -= cos(pitch) * cos(yaw) * camera_speed;
        dy -= cos(pitch) * sin(yaw) * camera_speed;
        dz -= sin(pitch) * camera_speed;
    }
    else if(key == 'd') {
        dx -= -sin(yaw) * camera_speed;
        dy -= cos(yaw) * camera_speed;
    }
}

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float lookat_x, lookat_y, lookat_z;
    camera_direction(&lookat_x, &lookat_y, &lookat_z);

    lookat_x += x;
    lookat_y += y;
    lookat_z += z;

    gluLookAt(x, y, z, lookat_x, lookat_y, lookat_z, 0.0f, 0.0f, 1.0f);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, aspect_ratio, 0.01f, 100.0f);

    glBindTexture(GL_TEXTURE_2D, particle_texture);

    glMatrixMode(GL_MODELVIEW);
    for(int i = 0; i < N_PARTICLES; ++i){
        glPushMatrix();

        glTranslatef(particle[i].x, particle[i].y, particle[i].z);

        glRotatef(particle[i].ax, 1.0f, 0.0f, 0.0f);
		glRotatef(particle[i].ay, 0.0f, 1.0f, 0.0f);
		glRotatef(particle[i].az, 0.0f, 0.0f, 1.0f);
        
        gluSphere(sphere, 0.5f, 20, 20);
        
        glPopMatrix();
    }

    for(int i = 0; i < N_MAX_BULLETS; ++i){
        if(!bullet_used_slot[i]){
            continue;
        }

        glPushMatrix();

        glTranslatef(bullet[i].x, bullet[i].y, bullet[i].z);

        gluSphere(sphere, 0.1f, 20, 20);
        
        glPopMatrix();
    }

    glBindTexture(GL_TEXTURE_2D, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glBindTexture(GL_TEXTURE_2D, floor_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, (GLsizei)floor_texture_size, (GLsizei)floor_texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
	// Последний параметр NULL в glTexImage2D говорит о том что данные для текстуры нужно брать из активного буфера
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-a + eps, -a + eps, 0.0);

		glTexCoord2f(1.0, 0.0);
		glVertex3f(a - eps, -a + eps, 0.0);

		glTexCoord2f(1.0, 1.0);
		glVertex3f(a - eps, a - eps, 0.0);

		glTexCoord2f(0.0, 1.0);
		glVertex3f(-a + eps, a - eps, 0.0);
	glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);

    glPushMatrix();

    glTranslatef(0, 0, a);
    glutWireCube(2 * a);

    glPopMatrix();

	glutSwapBuffers();
    glutPostRedisplay();

    glFlush();
}

void load_texture(const char *path, GLuint *texture){
    FILE *file = fopen(path, "r");

    int texture_w, texture_h;
    fread(&texture_w, sizeof(int), 1, file);
    fread(&texture_h, sizeof(int), 1, file);

    unsigned char *buffer = (unsigned char*)malloc(texture_w * texture_h * 4 * sizeof(unsigned char));
    fread(buffer, sizeof(unsigned char), texture_w * texture_h * 4, file);
    
    fclose(file);

    glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)texture_w, (GLsizei)texture_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*)buffer);

	// если полигон, на который наносим текстуру, меньше текстуры
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// если больше
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
    	
    free(buffer);
}


int main(int argc, char** argv) {
    
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(w, h);
	glutCreateWindow("OpenGL");

	glutIdleFunc(update);
    glutDisplayFunc(display);
    
    glutKeyboardFunc(keys);
    glutSetKeyRepeat(GLUT_KEY_REPEAT_DEFAULT);
    glutPassiveMotionFunc(mouse_movement);
	glutMouseFunc(mouse_clicks);
    
	const GLubyte *m_pVendor = glGetString(GL_VENDOR);
	const GLubyte *m_pRenderer = glGetString(GL_RENDERER);
    printf("vendor: %s\nrenderer: %s\n", (char*)m_pVendor, (char*)m_pRenderer);

    load_texture("particle.data", &particle_texture);

    glGenTextures(1, &floor_texture);

	glBindTexture(GL_TEXTURE_2D, floor_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glEnable(GL_TEXTURE_2D);                             // Разрешить наложение текстуры
	glShadeModel(GL_SMOOTH);                             // Разрешение сглаженного закрашивания
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);                // Черный фон
	glClearDepth(1.0f);                                  // Установка буфера глубины
	glDepthFunc(GL_LEQUAL);                              // Тип теста глубины. 
	glEnable(GL_DEPTH_TEST);                			 // Включаем тест глубины
	glEnable(GL_CULL_FACE);                 			 // Режим при котором, тектуры накладываются только с одной стороны

	glewInit();						

    glGenBuffers(1, &vbo);								// Получаем номер буфера
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);			// Делаем его активным
    glBufferData(GL_PIXEL_UNPACK_BUFFER, floor_texture_size * floor_texture_size * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);	// Задаем размер буфера
    
	CHECK_CUDA_CALL_ERROR(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard)); // Регистрируем буфер для использования его памяти в CUDA
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);			// Деактивируем буфер
    
	sphere = gluNewQuadric();
    gluQuadricDrawStyle(sphere, GLU_FILL);
    gluQuadricTexture(sphere, GL_TRUE);
    gluQuadricNormals(sphere, GLU_SMOOTH);

	glutSetCursor(GLUT_CURSOR_NONE);	// Скрываем курсор мышки
    glutWarpPointer(w / 2, h / 2);

    init();

    glutMainLoop();

    return 0;
}