#version 330 core
//http://outerra.blogspot.ca/2013/07/logarithmic-depth-buffer-optimizations.html
//fix same bugs with z-buffer and model killeroo
uniform vec4 color;
in float flogz;
in vec4 position;
const float far = 1000.0f;
const float Fcoef = 2.0 / log2(far + 1.0);
const float Fcoef_half = 0.5 * Fcoef;

out vec4 diffuseOut;

const float exposure = 0.5;
const float num = 100;

void main() {
    diffuseOut = position.zzzz;
    float f = 1.0f/1000.0f * (diffuseOut.x + diffuseOut.y + diffuseOut.z) / 3.0f;
    diffuseOut = vec4(f, f, f, 1.0f);
    const float gamma = 2.2;
    vec3 hdrColor = vec3(diffuseOut);
    vec3 mapped = vec3(1.0) - exp(-hdrColor * exposure);
    mapped = pow(mapped, vec3(1.0 / gamma));
    diffuseOut = vec4(mapped, 1.0);
    gl_FragDepth = log2(flogz) * Fcoef_half;
}
