#version 330 core

out vec4 color;

in vec3 magicPointPos;

varying vec3 norm;
varying vec3 pos;

uniform vec3 Ka;
uniform vec3 Kd;
uniform vec3 Ks;
uniform vec3 Ia;
uniform vec3 Il;
uniform float Ip;
uniform float K;

uniform float CameraPosition;
uniform vec3 LightPosition;

uniform vec3 objectColor;

uniform vec3 magicPointPosition;

varying float transparency;

uniform float magicCoef;
uniform bool distToFrag;

void main()
{
    // ambient
    vec3 AmbientComponent = Ka * Ia;

    // diffuse
    vec3 S = vec3(0.0, 0.0, CameraPosition) - pos;
    vec3 Snorm = normalize(S);

    float dist = length(S);

    vec3 L = normalize(LightPosition - pos);
    vec3 normal = normalize(norm);

    float LNcos = dot(L, normal);
    vec3 DiffuseComponent = (Kd * Il) / (K + dist) * clamp(LNcos, 0.0, 1.0);

    // specular
    vec3 R = reflect(-L, normal);

    float SRcos = dot(R, Snorm);
    vec3 SpecularComponent = Il * Ks * pow(clamp(SRcos, 0.0, 1.0), Ip) / (K + dist);
    vec3 result = AmbientComponent + DiffuseComponent + SpecularComponent;

    // shader effect
    if(!distToFrag){
		color = vec4(objectColor * result, transparency);
	}
    else{
        float magicDistance = distance(pos, magicPointPosition);
        color = vec4(objectColor * result, min(1.0, magicCoef / magicDistance));
    }
} 