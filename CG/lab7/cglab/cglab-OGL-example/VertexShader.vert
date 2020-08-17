#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;
layout(location = 2) in vec3 vertexColor;

// layout(location = 4) in mat4 projectionMatrix;
// layout(location = 5) in mat4 modelViewMatrix;

uniform modelViewMatrix;
uniform projectionMatrix;

out vc;

void main(){
	vc = vertexColor;
	
	gl_Position.xyz = projectionMatrix * modelViewMatrix * vertexPosition_modelspace;
	gl_Position.w = 1.0;
}