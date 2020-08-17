#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;

out vec3 magicPointPos;

uniform vec3 magicPointPosition;

varying vec3 norm;
varying vec3 pos;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 modelViewNormalMatrix;

uniform float magicCoef;
uniform bool distToFrag;

varying float transparency;

void main(){
	pos = (modelViewMatrix * vec4(vertexPosition_modelspace, 1.0)).xyz;
	gl_Position = (projectionMatrix * vec4(pos, 1.0));

	norm = (modelViewNormalMatrix * vec4(vertexNormal_modelspace, 1.0)).xyz;

	if(!distToFrag){
		transparency = min(1.0, magicCoef / distance(pos, magicPointPosition));
	}	
}