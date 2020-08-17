#version 330 core
in vec3 vc;

out vec4 color;

void main()
{
    color = vec4(vc, 1.0);
} 