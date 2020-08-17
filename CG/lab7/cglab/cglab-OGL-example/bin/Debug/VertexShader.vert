#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;
layout(location = 2) in vec3 vertexColor;

// layout(location = 2) uniform mat4 modelToWorldMatrix;

// layout(location = 3) uniform mat4 modelViewMatrix;
// layout(location = 4) uniform mat4 projectionMatrix;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

out vec3 vc;

void main(){
	vc = vertexColor;
	
	gl_Position = (projectionMatrix * modelViewMatrix * vec4(vertexPosition_modelspace, 1.0));

	vc = gl_Position.rgb;
	// gl_Position = vec4(vertexPosition_modelspace, 1.0);

}