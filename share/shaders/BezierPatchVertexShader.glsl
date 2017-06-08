#version 330 core
layout(location = 0) in vec4 vertex;
uniform mat4 MVP;
const float far = 1000.0f;
const float Fcoef = 2.0 / log2(far + 1.0);
out float flogz;
out vec4 position;
void main(){
    gl_Position = vertex;
    position = vertex;
    gl_Position.z = log2(max(1e-6, 1.0 + gl_Position.w)) * Fcoef - 1.0;
    flogz = 1.0 + gl_Position.w;
 }
